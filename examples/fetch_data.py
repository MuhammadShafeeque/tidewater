"""Fetch the data of the example glacier from the tidewater data home.

    python examples/fetch_data.py                    # the result files in examples/results/05-09345
    python examples/fetch_data.py inputs --to DIR    # the inputs of the host runs
    python examples/fetch_data.py --list

``data_files.csv`` lists every file with its set, size and sha256. A file that
is already there with the right checksum is not fetched again; a file with
another checksum is an error, never overwritten.
"""
import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = HERE / "data_files.csv"
URL = "https://cluster.klima.uni-bremen.de/~mshafeeque/tidewater/current/v1/"
TARGETS = {"results": HERE / "results" / "05-09345"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_files(path=FILES):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fetch(rows, to, url=URL):
    """Download ``rows`` into ``to``; returns the number of files fetched."""
    to, n = Path(to), 0
    for r in rows:
        dst = to / r["target"]
        if dst.exists():
            if sha256(dst) != r["sha256"]:
                raise ValueError(f"{dst} exists with another checksum; move it away first")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(url.rstrip("/") + "/" + r["path"], part)
        if sha256(part) != r["sha256"]:
            part.unlink()
            raise ValueError(f"{r['path']}: the checksum of the download is not the listed one")
        part.rename(dst)
        n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("set", nargs="?", default="results", help="results (default) or inputs")
    p.add_argument("--to", help="directory to write to; needed for every set but results")
    p.add_argument("--list", action="store_true", help="print the file list and fetch nothing")
    p.add_argument("--url", default=URL, help="base address of the data home")
    a = p.parse_args(argv)
    rows = [r for r in read_files() if a.list or r["set"] == a.set]
    if a.list:
        for r in rows:
            print(f"{r['set']:8s} {int(r['size']):9d}  {r['path']}")
        return 0
    if not rows:
        p.error(f"no set {a.set!r} in {FILES.name}")
    to = a.to or TARGETS.get(a.set) or p.error("--to is needed for this set")
    n = fetch(rows, to, a.url)
    print(f"{len(rows)} files in {to}, {n} fetched, checksums equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
