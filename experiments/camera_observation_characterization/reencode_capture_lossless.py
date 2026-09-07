#!/usr/bin/env python3
"""Journaled lossless PNG-to-WebP conversion of a stopped, running capture.

Each image's new bytes and index are durable before removing its original. A retry
verifies the journal and recovers any interrupted step. Requires space for one new
encoded image, the journal and one index replacement; it never overwrites an inode.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/unav_common'))
from unav_common.capture_integrity import (
    atomic_bytes, atomic_csv, atomic_json, capture_lock, capture_path, checked_image, pixel_hash,
)
sha1_array = pixel_hash

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(capture: Path) -> dict:
    capture = Path(capture).resolve()
    with capture_lock(capture):
        index = capture / 'capture_index.csv'
        manifest_path = capture / 'capture_manifest.json'
        journal_path = capture / 'storage_conversion.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'running':
            raise RuntimeError('only status=running captures may be converted')
        with index.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError('capture index is empty')
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if journal.get('schema') != 'capture_storage_conversion.v1':
                raise RuntimeError('unknown storage conversion journal')
            original = journal['rows']
            updated = dict(journal['manifest'], storage_reencoding=journal.get('summary'))
            if manifest not in (journal['manifest'], updated):
                raise RuntimeError('capture manifest changed during conversion')
        else:
            original = rows
            operations = {}
            for row in rows:
                if row['capture_status'] != 'ok' or not row['image'].endswith('.png'):
                    continue
                source = capture_path(capture, row['image'])
                target = source.with_suffix('.webp')
                op = dict(source=row['image'], target=str(target.relative_to(capture)),
                          image_sha1=row['image_sha1'], bytes_before=source.stat().st_size)
                if row['image'] in operations and operations[row['image']]['image_sha1'] != row['image_sha1']:
                    raise RuntimeError('one path has multiple image identities')
                checked_image(capture, row)
                operations[row['image']] = op
            if not operations:
                raise RuntimeError('no PNG images to convert')
            journal = dict(schema='capture_storage_conversion.v1', rows=rows, manifest=manifest,
                           operations=list(operations.values()), complete=False)
            atomic_json(journal_path, journal)
        mapping = {op['source']: op['target'] for op in journal['operations']}
        if len(rows) != len(original):
            raise RuntimeError('capture row count changed during conversion')
        for old, row in zip(original, rows, strict=True):
            expected = dict(old, image=row['image'])
            if row != expected or row['image'] not in (old['image'], mapping.get(old['image'])):
                raise RuntimeError('capture index changed outside the conversion transaction')
        for op in journal['operations']:
            source = capture_path(capture, op['source'])
            target = capture_path(capture, op['target'])
            source_row = dict(image=op['source'], image_sha1=op['image_sha1'])
            target_row = dict(image=op['target'], image_sha1=op['image_sha1'])
            if source.exists():
                image = checked_image(capture, source_row)
            elif any(r['image'] == op['source'] for r in rows):
                raise RuntimeError(f'indexed original is missing: {source}')
            if target.exists():
                checked_image(capture, target_row)
            else:
                if not source.exists():
                    raise RuntimeError(f'original and converted image are missing: {source}')
                ok, encoded = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 101])
                if not ok:
                    raise RuntimeError(f'lossless WebP encode failed: {source}')
                verify = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)
                if verify is None or pixel_hash(verify) != op['image_sha1']:
                    raise RuntimeError(f'lossless codec changed pixels: {source}')
                atomic_bytes(target, encoded.tobytes())
                checked_image(capture, target_row)
            changed = [dict(row, image=op['target']) if row['image'] == op['source'] else row
                       for row in rows]
            if changed != rows:
                atomic_csv(index, changed, rows[0].keys())
                rows = changed
            # Index replacement (including directory fsync) committed before deletion.
            if source.exists():
                source.unlink()
        if 'summary' not in journal:
            before = sum(op['bytes_before'] for op in journal['operations'])
            after = sum(capture_path(capture, op['target']).stat().st_size for op in journal['operations'])
            journal['summary'] = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                operation='journaled lossless PNG to WebP; decoded BGR arrays unchanged',
                utility_sha256=sha256(Path(__file__)), files_reencoded=len(mapping),
                bytes_before=before, bytes_after=after, bytes_freed=before-after,
                decoded_hashes_verified_before_and_after=True)
            atomic_json(journal_path, journal)
        atomic_json(manifest_path, dict(journal['manifest'], storage_reencoding=journal['summary']))
        journal['complete'] = True
        atomic_json(journal_path, journal)
        return journal['summary']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.capture), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
