"""
Utilities to convert MCS .rda result files to CSV and merge them, without rpy2 or pandas.

This script relies on `Rscript` being available on PATH. For each .rda file it
loads the object(s), grabs an `effect` column (or, failing that, the first vector),
and writes a one-column CSV. It then merges those CSVs horizontally into
`merged_csvs.csv`.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List


def convert_rda_to_csv(rda_dir: Path, csv_dir: Path) -> List[Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    for rda_path in sorted(rda_dir.glob("*.rda")):
        out_path = csv_dir / f"{rda_path.stem}.csv"
        r_code = f"""
in_path <- "{rda_path}"
out_path <- "{out_path}"
loaded_names <- load(in_path)
effect_col <- NULL
for (nm in loaded_names) {{
  obj <- get(nm)
  if (is.data.frame(obj) && "effect" %in% names(obj)) {{
    effect_col <- obj$effect
    break
  }}
  if (is.list(obj) && !is.null(obj$effect)) {{
    effect_col <- obj$effect
    break
  }}
  if (is.vector(obj) || is.matrix(obj)) {{
    effect_col <- obj
    break
  }}
}}
if (is.null(effect_col)) stop("no effect column found")
write.csv(effect_col, out_path, row.names = FALSE)
"""
        try:
            subprocess.run(["Rscript", "-e", r_code], check=True, capture_output=True)
            if out_path.exists():
                outputs.append(out_path)
                print(f"[convert] wrote {out_path.name} via Rscript")
            else:
                print(f"[convert] Rscript ran but {out_path.name} missing", file=sys.stderr)
        except FileNotFoundError:
            print(f"[convert] Skipping {rda_path.name}: Rscript not found", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
            print(f"[convert] Skipping {rda_path.name}: Rscript failed ({stderr.strip()})", file=sys.stderr)
    return outputs


def merge_csvs(csv_dir: Path, merged_dir: Path, output_name: str = "merged_csvs.csv") -> Path | None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print("[merge] No CSV files found; skipping merge", file=sys.stderr)
        return None

    # Read each CSV as a list of strings (one column)
    columns: List[List[str]] = []
    headers: List[str] = []
    for csv_path in csv_files:
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            rows = [row[0] for row in reader if row]
        # drop header if present
        if rows and rows[0].lower() == "x":
            rows = rows[1:]
        columns.append(rows)
        headers.append(csv_path.stem)

    # Pad shorter columns with empty strings if needed
    max_len = max(len(col) for col in columns)
    for col in columns:
        if len(col) < max_len:
            col.extend([""] * (max_len - len(col)))

    out_path = merged_dir / output_name
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row_idx in range(max_len):
            writer.writerow([col[row_idx] for col in columns])
    return out_path


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent / ".." / "data" / "mcs" / "results"
    parser = argparse.ArgumentParser(
        description="Convert MCS .rda outputs to CSV and merge them (uses Rscript only)."
    )
    parser.add_argument("--rda-dir", type=Path, default=base, help="Directory containing .rda files.")
    parser.add_argument("--csv-dir", type=Path, default=base / "csvs", help="Directory to write CSVs.")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=base / "merged_files",
        help="Directory to write the merged CSV.",
    )
    parser.add_argument("--skip-convert", action="store_true", help="Skip .rda -> CSV conversion.")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merging CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted: List[Path] = []
    if not args.skip_convert:
        converted = convert_rda_to_csv(args.rda_dir, args.csv_dir)
        print(f"[convert] finished: {len(converted)} file(s) written to {args.csv_dir}")
    if not args.skip_merge:
        merged = merge_csvs(args.csv_dir, args.merged_dir)
        if merged:
            print(f"[merge] wrote {merged.name} in {args.merged_dir}")


if __name__ == "__main__":
    main()
