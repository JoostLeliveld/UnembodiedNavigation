#!/usr/bin/env python3
"""Fetch the frozen TorWIC RGB-D subset without downloading its 3.75 GB ZIP.

Google Drive serves public files with byte ranges.  This script reads the ZIP's
end-of-central-directory record, fetches its directory, and then requests only
the 48 members frozen in ``PREREGISTRATION.md``.  Every decompressed member is
checked against the ZIP size and CRC before it is kept.

The downloaded images remain subject to TorWIC's CC BY-NC 4.0 plus additional
dataset terms: https://github.com/Viky397/TorWICDataset/tree/main/License
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib


ARCHIVE_ID = "1hplx0_5tDKz4zF6iRgTfwuQNIund0URn"
ARCHIVE_NAME = "Aisle_CCW.zip"
ARCHIVE_SIZE = 3_753_328_323
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={ARCHIVE_ID}&export=download&confirm=t"
)
CALIBRATION_ID = "1NVnNEi-9QDoeyrnkxtlv8dHZl4Sc79zw"
CALIBRATION_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={CALIBRATION_ID}&export=download&confirm=t"
)
FRAMES = ("000000", "000115", "000230", "000345", "000460", "000575", "000690", "000805")
SIDES = ("left", "right")
MODALITIES = ("image", "depth", "segmentation_greyscale")
ROOT = "Aisle_CCW"
TAIL_BYTES = 65_557
LOCAL_EXTRA_BUDGET = 4_096


@dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int
    crc32: int


def _url_bytes(url: str, *, byte_range: tuple[int, int] | None = None, retries: int = 5) -> bytes:
    headers = {"User-Agent": "UnembodiedNavigation-TorWIC-sanity/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                data = response.read()
                if byte_range is not None:
                    expected = byte_range[1] - byte_range[0] + 1
                    if response.status != 206 or len(data) != expected:
                        raise RuntimeError(
                            f"range {byte_range} returned HTTP {response.status}, {len(data)} bytes"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    suffix = f"/{ARCHIVE_SIZE}"
                    if not content_range.endswith(suffix):
                        raise RuntimeError(
                            f"archive size changed or missing Content-Range: {content_range!r}"
                        )
                return data
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}") from last


def _central_directory() -> dict[str, ZipMember]:
    tail_start = ARCHIVE_SIZE - TAIL_BYTES
    tail = _url_bytes(ARCHIVE_URL, byte_range=(tail_start, ARCHIVE_SIZE - 1))
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    fields = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    _, disk, cdir_disk, n_disk, n_total, cdir_size, cdir_offset, comment_len = fields
    if disk or cdir_disk or n_disk != n_total or comment_len:
        raise RuntimeError(f"unsupported multi-disk/commented ZIP EOCD: {fields}")
    cdir = _url_bytes(
        ARCHIVE_URL,
        byte_range=(cdir_offset, cdir_offset + cdir_size - 1),
    )
    members: dict[str, ZipMember] = {}
    offset = 0
    central_fmt = "<4s6H3L5H2L"
    while offset + 46 <= len(cdir):
        fields = struct.unpack_from(central_fmt, cdir, offset)
        if fields[0] != b"PK\x01\x02":
            break
        (
            _, _made, _needed, _flags, method, _time, _date, crc, compressed,
            uncompressed, name_len, extra_len, comment_len, _disk, _int_attr,
            _ext_attr, local_offset,
        ) = fields
        name_b = cdir[offset + 46 : offset + 46 + name_len]
        name = name_b.decode("utf-8")
        members[name] = ZipMember(
            name=name,
            method=int(method),
            compressed_size=int(compressed),
            uncompressed_size=int(uncompressed),
            local_offset=int(local_offset),
            crc32=int(crc),
        )
        offset += 46 + name_len + extra_len + comment_len
    if offset != len(cdir) or len(members) != n_total:
        raise RuntimeError(
            f"central directory parsed {len(members)}/{n_total} members and "
            f"{offset}/{len(cdir)} bytes"
        )
    return members


def _selected_names() -> list[str]:
    return [
        f"{ROOT}/{modality}_{side}/{frame}.png"
        for side in SIDES
        for modality in MODALITIES
        for frame in FRAMES
    ]


def _valid_existing(path: Path, member: ZipMember) -> bool:
    if not path.is_file() or path.stat().st_size != member.uncompressed_size:
        return False
    return (zlib.crc32(path.read_bytes()) & 0xFFFFFFFF) == member.crc32


def _extract_member(member: ZipMember, out_root: Path, force: bool) -> dict:
    relative = PurePosixPath(member.name).relative_to(ROOT)
    if any(part in ("", ".", "..") for part in relative.parts):
        raise RuntimeError(f"unsafe ZIP member path: {member.name!r}")
    destination = out_root.joinpath(*relative.parts)
    if not force and _valid_existing(destination, member):
        payload = destination.read_bytes()
        return {
            **asdict(member),
            "path": str(destination.relative_to(out_root.parent)),
            "sha256": sha256(payload).hexdigest(),
            "downloaded": False,
        }

    # Fetch local header and payload in one request.  The 4 KiB allowance is
    # far larger than these entries' local extra fields and avoids a second
    # HTTP round trip per member.
    start = member.local_offset
    provisional_end = (
        start + 30 + len(member.name.encode("utf-8"))
        + LOCAL_EXTRA_BUDGET + member.compressed_size
    )
    blob = _url_bytes(ARCHIVE_URL, byte_range=(start, provisional_end - 1))
    local = struct.unpack_from("<4s5H3L2H", blob, 0)
    if local[0] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {member.name}")
    method, name_len, extra_len = int(local[3]), int(local[9]), int(local[10])
    if method != member.method or extra_len > LOCAL_EXTRA_BUDGET:
        raise RuntimeError(
            f"unexpected local header for {member.name}: method={method}, extra={extra_len}"
        )
    local_name = blob[30 : 30 + name_len].decode("utf-8")
    if local_name != member.name:
        raise RuntimeError(f"central/local name mismatch: {member.name!r} vs {local_name!r}")
    data_at = 30 + name_len + extra_len
    compressed = blob[data_at : data_at + member.compressed_size]
    if len(compressed) != member.compressed_size:
        raise RuntimeError(f"truncated compressed payload for {member.name}")
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported ZIP method {method} for {member.name}")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    if len(payload) != member.uncompressed_size or crc != member.crc32:
        raise RuntimeError(
            f"integrity failure for {member.name}: size {len(payload)}/{member.uncompressed_size}, "
            f"CRC {crc:08x}/{member.crc32:08x}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return {
        **asdict(member),
        "path": str(destination.relative_to(out_root.parent)),
        "sha256": sha256(payload).hexdigest(),
        "downloaded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = (
        Path(__file__).resolve().parents[3]
        / "logs/studies/reconfiguration_holdout/real_rgbd_sanity/torwic_subset"
    )
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw"

    members = _central_directory()
    selected = _selected_names()
    missing = sorted(set(selected) - set(members))
    if missing:
        raise RuntimeError(f"frozen members missing from archive: {missing}")

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_extract_member, members[name], raw, args.force): name
            for name in selected
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            action = "downloaded" if record["downloaded"] else "verified"
            print(f"[{index:02d}/{len(futures)}] {action}: {futures[future]}", flush=True)

    calibration = _url_bytes(CALIBRATION_URL)
    calibration_path = args.out / "calibrations.txt"
    calibration_path.write_bytes(calibration)
    manifest = {
        "protocol": "experiments/reconfiguration_holdout/real_rgbd_sanity/PREREGISTRATION.md",
        "dataset": "Toronto Warehouse Incremental Change (TorWIC)-SLAM",
        "official_repository": "https://github.com/Viky397/TorWICDataset",
        "license_terms": "https://github.com/Viky397/TorWICDataset/tree/main/License",
        "archive": {
            "name": ARCHIVE_NAME,
            "google_drive_file_id": ARCHIVE_ID,
            "advertised_size_bytes": ARCHIVE_SIZE,
            "member_count_in_archive": len(members),
            "download_method": "HTTP byte-range extraction; complete archive not downloaded",
        },
        "calibration": {
            "google_drive_file_id": CALIBRATION_ID,
            "path": calibration_path.name,
            "size_bytes": len(calibration),
            "sha256": sha256(calibration).hexdigest(),
        },
        "commissioning_frame": FRAMES[0],
        "test_frames": list(FRAMES[1:]),
        "cameras": list(SIDES),
        "members": sorted(records, key=lambda item: item["name"]),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
