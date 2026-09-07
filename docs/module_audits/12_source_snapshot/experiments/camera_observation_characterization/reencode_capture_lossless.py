#!/usr/bin/env python3
"""Losslessly re-encode capture PNGs as WebP while preserving decoded image hashes.

This is an emergency storage operation, not an image perturbation. Every image is decoded,
checked against the capture-time array SHA1, encoded with OpenCV's lossless WebP mode,
decoded again, and checked before its index path is changed. Pixel arrays are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]


def sha1_array(image) -> str:
    digest = hashlib.sha1()  # nosec B324 - provenance, not security
    digest.update(str(image.shape).encode('ascii'))
    digest.update(str(image.dtype).encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    index = capture / 'capture_index.csv'
    manifest_path = capture / 'capture_manifest.json'
    rows = list(csv.DictReader(index.open(newline='', encoding='utf-8')))
    if not rows:
        raise RuntimeError('capture index is empty')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'running':
        raise RuntimeError('this recovery tool only accepts status=running captures')

    expected_by_path: dict[str, set[str]] = {}
    for row in rows:
        if row.get('capture_status') == 'ok' and row.get('image'):
            expected_by_path.setdefault(row['image'], set()).add(row['image_sha1'])
    if any(len(values) != 1 for values in expected_by_path.values()):
        raise RuntimeError('one stored path is associated with multiple decoded hashes')

    mapping = {}
    before = after = 0
    for number, relative in enumerate(sorted(expected_by_path), 1):
        source = capture / relative
        if source.suffix.lower() != '.png':
            continue
        target = source.with_suffix('.webp')
        if target.exists():
            raise RuntimeError(f'target already exists: {target}')
        encoded_bytes = source.stat().st_size
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        expected = next(iter(expected_by_path[relative]))
        if image is None or sha1_array(image) != expected:
            raise RuntimeError(f'pre-encode decoded hash mismatch: {source}')
        ok, encoded = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 101])
        if not ok:
            raise RuntimeError(f'lossless WebP encode failed: {source}')

        # Overwrite the existing inode first. This needs no second image-sized allocation,
        # which is important when this utility is invoked because the filesystem is full.
        source.write_bytes(encoded.tobytes())
        verify = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if verify is None or sha1_array(verify) != expected:
            raise RuntimeError(f'post-encode decoded hash mismatch: {source}')
        source.rename(target)
        mapping[relative] = str(target.relative_to(capture))
        before += encoded_bytes
        after += target.stat().st_size
        if number % 256 == 0 or number == len(expected_by_path):
            print(f're-encoded {number}/{len(expected_by_path)} unique files', flush=True)

    if not mapping:
        raise RuntimeError('no PNG images were found to re-encode')
    for row in rows:
        if row.get('image') in mapping:
            row['image'] = mapping[row['image']]
    temporary = index.with_suffix('.csv.reencode_tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(index)

    manifest['storage_reencoding'] = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'operation': 'lossless PNG to lossless WebP; decoded BGR arrays unchanged',
        'utility': str(Path(__file__).resolve()),
        'utility_sha256': sha256(Path(__file__).resolve()),
        'files_reencoded': len(mapping),
        'bytes_before': before,
        'bytes_after': after,
        'bytes_freed': before - after,
        'decoded_hashes_verified_before_and_after': True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['storage_reencoding'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
