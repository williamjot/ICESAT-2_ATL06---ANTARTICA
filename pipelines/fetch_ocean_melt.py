"""Baixa uma única vez e valida o pequeno conjunto BAS/ITGC MELT."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites.ocean.bas_melt import (  # noqa: E402
    BAS_MELT_DOI,
    BAS_MELT_SHA256,
    BAS_MELT_SIZE,
    file_sha256,
    validate_bas_melt_file,
)

ENTRY_ID = (
    "synth:4ffad557-1c3c-4ea7-a73d-6d782331b08a:"
    "L1Rod2FpdGVzX01BVlNfVGltZXNlcmllc19UU1YuZGF0"
)
URL = "https://ramadda.data.bas.ac.uk/repository/entry/get?entryid=" + urllib.parse.quote(ENTRY_ID)
FILENAME = "Thwaites_MAVS_Timeseries_TSV.dat"


def main():
    parser = argparse.ArgumentParser(description="BAS/ITGC MELT 01468.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "ocean" / "bas_melt_01468"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / FILENAME
    if destination.exists() and not args.force:
        validate_bas_melt_file(destination)
        print(f"cache válido: {destination}")
    else:
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "thwaites-icesat2/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if partial.stat().st_size != BAS_MELT_SIZE:
                raise ValueError(
                    f"download incompleto: {partial.stat().st_size} bytes; esperado {BAS_MELT_SIZE}")
            if file_sha256(partial) != BAS_MELT_SHA256:
                raise ValueError("SHA256 do download não corresponde ao registro validado")
            os.replace(partial, destination)
        finally:
            if partial.exists():
                partial.unlink()
        print(f"baixado e validado: {destination}")

    metadata = {
        "dataset": "BAS/ITGC MELT 01468",
        "doi": BAS_MELT_DOI,
        "source_url": URL,
        "filename": FILENAME,
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "Open Government Licence v3.0",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
