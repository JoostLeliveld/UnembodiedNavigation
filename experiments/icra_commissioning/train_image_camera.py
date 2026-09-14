"""Train a genuine RGB residual/covariance model on the existing spatial split.

Development experiment, static commanded-pose reference. No sequential claim.
Usage: python3 experiments/icra_commissioning/train_image_camera.py --out PATH
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/p) for p in ('src/unav_common', 'src/experiments', 'src/planning')]
from study import REPO, OUT, load, readcsv, digest, score
from reliability.learned_box_correction import LearnedBoxCorrection
from reliability.image_camera import ImageCameraNet, paired_crop, gaussian_nll


def group_weights(groups):
    return np.array([1 / groups.count(g) for g in groups], dtype='float32')


def prepare(out):
    manifest = json.loads((OUT/'manifest.json').read_text())
    data, counts = load(OUT, write_records=False)
    base = LearnedBoxCorrection(REPO/manifest['mean_artifact'])
    rows = {(r['camera_id'], f"{r['pose_id']}:{r['repetition_id']}"): r
            for r in readcsv(REPO/manifest['capture']/'bias_update_interpretations.csv')}
    features, targets, images, bases = [], [], [], []
    for i, r in enumerate(data):
        source = rows[r['camera'], r['frame']]
        box = [float(source[k]) for k in ('x0','y0','x1','y1')]
        image_path = REPO/manifest['capture']/source['image']
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(image_path)
        features.append(base._features(r['camera'], r['raw'], box, r['confidence']))
        images.append(paired_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box))
        bases.append(r['basis'])
        targets.append(r['basis'].T @ (r['truth']-r['z']))
        if i % 1000 == 0:
            print(f'Prepared {i}/{len(data)} RGB pairs', flush=True)
    arrays = dict(features=np.asarray(features, dtype='float32'),
        targets=np.asarray(targets, dtype='float32'), images=np.asarray(images),
        bases=np.asarray(bases), role=np.array([r['role'] for r in data]),
        group=np.array([r['group'] for r in data]), camera=np.array([r['camera'] for r in data]))
    np.savez_compressed(out/'training_arrays.npz', **arrays)
    (out/'inputs.json').write_text(json.dumps(dict(manifest_sha256=digest(OUT/'manifest.json'),
        base_sha256=base.sha256, counts=counts, status='static_development_only'), indent=2))
    return arrays, base


def fit(a, base, out, use_image, epochs):
    torch.manual_seed(709)
    roles = a['role']
    train = np.flatnonzero(roles == 'mean_train')
    covfit = np.flatnonzero(roles == 'covariance_fit')
    select = np.flatnonzero(roles == 'selection')
    evaluate = np.flatnonzero(roles == 'evaluation')
    centre, spread = a['features'][train].mean(0), a['features'][train].std(0).clip(.01)
    X = torch.from_numpy((a['features']-centre)/spread)
    images, target = torch.from_numpy(a['images']), torch.from_numpy(a['targets'])
    model = ImageCameraNet(X.shape[1], use_image)
    name = 'rgb' if use_image else 'scalar'
    history = []
    for stage, indices in [('mean', train), ('covariance', covfit)]:
        for p in model.parameters():
            p.requires_grad_(stage == 'mean')
        for p in model.cov_head.parameters():
            p.requires_grad_(stage == 'covariance')
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        weights = torch.from_numpy(group_weights(a['group'][indices].tolist()))
        weights /= weights.mean()
        best, best_state = np.inf, copy.deepcopy(model.state_dict())
        rng = np.random.default_rng(709)
        selection_weights = torch.from_numpy(group_weights(a['group'][select].tolist()))
        for epoch in range(epochs):
            model.train()
            for local in np.array_split(rng.permutation(len(indices)), max(1, int(np.ceil(len(indices)/128)))):
                batch = indices[local]
                mean, L = model(images[batch], X[batch])
                loss = (mean-target[batch]).square().sum(-1) if stage == 'mean' else gaussian_nll(target[batch], mean, L)
                loss = (loss*weights[local]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                mean, L = model(images[select], X[select])
                losses = (mean-target[select]).square().sum(-1) if stage == 'mean' else gaussian_nll(target[select], mean, L)
                validation = float((losses*selection_weights).sum()/selection_weights.sum())
            history.append(dict(stage=stage, epoch=epoch, selection_loss=validation))
            if validation < best:
                best, best_state = validation, copy.deepcopy(model.state_dict())
            if epoch % 5 == 0:
                print(name, stage, epoch, validation, flush=True)
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mean, L = model(images, X)
    mean, L = mean.numpy(), L.numpy()
    residual = mean-a['targets']
    R = L @ L.transpose(0, 2, 1)
    white = np.linalg.solve(L[select], residual[select, :, None])[..., 0]
    groups = a['group'][select]
    scale = float(np.mean([np.mean(np.sum(white[groups == g]**2, axis=1))/2 for g in set(groups)]))
    scale = max(scale, .01)
    eworld = np.einsum('nij,nj->ni', a['bases'], residual)
    Rworld = a['bases'] @ (R*scale) @ a['bases'].transpose(0, 2, 1)
    results = score(eworld[evaluate], Rworld[evaluate], a['group'][evaluate])
    results['per_camera'] = {c: score(eworld[idx], Rworld[idx], a['group'][idx])
        for c in sorted(set(a['camera']))
        for idx in [evaluate[a['camera'][evaluate] == c]]}
    results['covariance_scale'] = scale
    torch.save(dict(schema='image_camera.v1', base_sha256=base.sha256,
        use_image=use_image, feature_mean=centre.tolist(), feature_std=spread.tolist(),
        covariance_scale=scale, state_dict=model.state_dict(),
        target='truth minus existing NN in along/across ray metres',
        status='static_development_only'), out/f'{name}.pt')
    # Paired group effects can be reconstructed without another image pass.
    np.savez_compressed(out/f'{name}_evaluation.npz', error=eworld[evaluate],
        covariance=Rworld[evaluate], group=a['group'][evaluate], camera=a['camera'][evaluate])
    (out/f'{name}_history.json').write_text(json.dumps(history, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    a, base = prepare(args.out)
    results = {name: fit(a, base, args.out, use_image, args.epochs)
               for name, use_image in [('scalar', False), ('rgb', True)]}
    # Retain the actual existing deployed mean + commissioned full covariance baseline.
    import joblib
    data, _ = load(OUT, write_records=False)
    models = joblib.load(OUT/'models.joblib')
    indices = np.flatnonzero(a['role'] == 'evaluation')
    errors, covs = [], []
    for i in indices:
        r = data[i]
        z, R = models[r['camera'], 'constant'].predict([r])
        errors.append(z[0]-r['truth'])
        covs.append(R[0])
    results['existing_nn_constant'] = score(errors, covs, a['group'][indices])
    results['scope'] = 'Current camera readings; static commanded-pose reference; existing grouped development split; no driving, temporal or final ICRA result.'
    results['counts_by_role'] = {r: int(sum(a['role'] == r)) for r in sorted(set(a['role']))}
    results['epochs_per_stage'] = args.epochs
    results['source_sha256'] = {str(p.relative_to(REPO)): digest(p) for p in [Path(__file__),
        REPO/'src/reliability/reliability/image_camera.py']}
    (args.out/'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
