"""Download Gset Max-Cut instances into ./instance/Gset/ as G{id}.txt.

The Gset benchmark (Yinyu Ye, Stanford) ships each graph as a plain text file whose
first line is "num_nodes num_edges" followed by "u v w" edge lines. parse_gset expects
exactly this format under ./instance/Gset/G{id}.txt, so we save with the .txt suffix.

Size classes used by the sweep (representative instances, paper Table 11):
    small  : G1   (800 nodes,   19176 edges)
    medium : G22  (2000 nodes,  19990 edges)
    large  : G67  (10000 nodes, 20000 edges)
    xlarge : G81  (20000 nodes, 40000 edges)   # optional, heaviest

Run this ON the machine that will do the sweep (e.g. bhz), which needs outbound
internet to Stanford. Example:
    python scripts/download_gset.py --ids 1 22 67 81
    python scripts/download_gset.py --preset small medium large
"""
import argparse
import os
import sys
import urllib.request

BASE_URL = "https://web.stanford.edu/~yyye/yyye/Gset/G{id}"

PRESETS = {
    "small": [1],
    "medium": [22],
    "large": [67],
    "xlarge": [81],
}


def build_parser():
    p = argparse.ArgumentParser(description="Download Gset instances for the Max-Cut sweep.")
    p.add_argument("--ids", type=int, nargs="+", default=None,
                   help="explicit Gset ids to fetch, e.g. --ids 1 22 67 81")
    p.add_argument("--preset", nargs="+", choices=list(PRESETS), default=None,
                   help="size presets to fetch (small=G1, medium=G22, large=G67, xlarge=G81)")
    p.add_argument("--base_url", default=BASE_URL,
                   help="URL template with a {id} placeholder (change if the mirror moves)")
    p.add_argument("--dest", default="./instance/Gset", help="output directory")
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    return p


def main():
    args = build_parser().parse_args()

    ids = []
    if args.preset:
        for name in args.preset:
            ids.extend(PRESETS[name])
    if args.ids:
        ids.extend(args.ids)
    if not ids:
        ids = [1, 22, 67, 81]  # default: one per size class + xlarge
    ids = sorted(set(ids))

    os.makedirs(args.dest, exist_ok=True)
    ok, failed = [], []
    for gid in ids:
        out_path = os.path.join(args.dest, f"G{gid}.txt")
        if os.path.exists(out_path) and not args.force:
            print(f"[keep] G{gid}.txt already exists (use --force to overwrite)")
            ok.append(gid)
            continue
        url = args.base_url.format(id=gid)
        try:
            print(f"[get ] G{gid}  <- {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                content = resp.read()
            # sanity check: first line should parse as "n m"
            first = content.split(b"\n", 1)[0].split()
            if len(first) != 2 or not all(tok.isdigit() for tok in first):
                raise ValueError(f"unexpected first line: {first!r} (not 'n m')")
            with open(out_path, "wb") as f:
                f.write(content)
            n, m = int(first[0]), int(first[1])
            print(f"       saved {out_path}  (n={n}, m={m})")
            ok.append(gid)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[FAIL] G{gid}: {exc}", file=sys.stderr)
            failed.append(gid)

    print(f"\nDone. ok={['G%d' % g for g in ok]}  failed={['G%d' % g for g in failed]}")
    if failed:
        print("If downloads fail (no internet on this host / mirror moved), fetch the files "
              "manually and place them as ./instance/Gset/G<id>.txt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
