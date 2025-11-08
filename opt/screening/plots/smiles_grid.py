#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
smiles_grid.py — Generate SVGs for SMILES with Open Babel and collate into a numbered SVG grid.

Usage examples:
  # From a text file with one SMILES per line
  ./smiles_grid.py --out-dir svgs --cols 4 smiles.txt grid.svg

Requires:
  - Open Babel CLI (`obabel`) on PATH.

Notes:
  - If both --rows and --cols are provided and the number of molecules exceeds rows*cols,
    output is split across multiple SVGs named like <stem>_p01<suffix>, <stem>_p02<suffix>, etc.
"""

import argparse
import base64
import math
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def run_obabel_to_svg(smiles: str, out_svg: Path, obabel: str = "obabel") -> None:
    """
    Use Open Babel CLI to render a SMILES string to an SVG file.

    We generate 2D coordinates and write SVG. The inline SMILES form ( -:"SMI" ) avoids temp files.
    """
    # Ensure parent exists
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    # Command:
    #   obabel -:"<SMILES>" -O out.svg --gen2D
    # Notes:
    # - `--gen2D` asks for 2D depiction (Open Babel will generate coordinates if missing).
    # - We add `-d` to drop explicit hydrogens (cleaner drawing); remove if you want Hs.
    cmd = [obabel, f'-:"{smiles}"', "-O", str(out_svg), "--gen2D", "-d"]

    # Use shell=False; the -:"..." form must be a single arg, we already quoted inside.
    # To keep it robust across shells, pass via stdin instead (avoids quoting pain).
    # Alternative robust approach:
    cmd = [obabel, "-ismi", "-osvg", "-O", str(out_svg), "-d", "--gen2D"]

    try:
        proc = subprocess.run(
            cmd,
            input=(smiles + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Open Babel CLI 'obabel' not found on PATH. Please install Open Babel and ensure 'obabel' is available."
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Open Babel failed for SMILES: {smiles}\nSTDOUT:\n{proc.stdout.decode()}\nSTDERR:\n{proc.stderr.decode()}"
        )

    if not out_svg.exists() or out_svg.stat().st_size == 0:
        raise RuntimeError(f"Open Babel did not produce an SVG for: {smiles}")


def load_smiles_from_file(path: Path) -> list[str]:
    """
    Reads SMILES from a text file (one per line). Lines may contain comments after whitespace '#'.
    Blank lines are ignored.
    """
    smiles: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        smiles.append(line)
    return smiles


def infer_grid(n: int, rows: int | None, cols: int | None) -> tuple[int, int]:
    if rows is not None and cols is not None:
        return rows, cols
    if rows is not None:
        cols = math.ceil(n / rows)
        return rows, cols
    if cols is not None:
        rows = math.ceil(n / cols)
        return rows, cols
    # Default: make a roughly square grid
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def svg_data_uri(svg_text: str) -> str:
    """
    Embed an SVG as a data URI for <image href="...">. We base64-encode to avoid escaping issues.
    """
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _extract_svg_inner(
    svg_text: str,
) -> tuple[str | None, float | None, float | None, str]:
    """
    Extract inner SVG content and attributes from a child SVG string.

    Returns (viewBox, width_px, height_px, inner_content)
    - viewBox: raw viewBox string if present on root <svg>, else None
    - width_px/height_px: numeric width/height if present on root <svg>, else None
    - inner_content: content between the opening <svg ...> and closing </svg>
    """
    import re

    # Strip XML declaration/doctype which are invalid inside an embedded SVG
    lines = [
        ln
        for ln in svg_text.splitlines()
        if not ln.strip().startswith(("<?xml", "<!DOCTYPE"))
    ]
    text = "\n".join(lines)

    m = re.search(r"<svg\b([^>]*)>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None, None, text
    attrs = m.group(1)
    start = m.end()
    end = text.lower().rfind("</svg>")
    if end == -1:
        end = len(text)
    inner = text[start:end]

    def _attr(name: str) -> str | None:
        mm = re.search(rf"{name}\s*=\s*\"([^\"]+)\"", attrs, flags=re.IGNORECASE)
        if mm:
            return mm.group(1)
        mm = re.search(rf"{name}\s*=\s*'([^']+)'", attrs, flags=re.IGNORECASE)
        return mm.group(1) if mm else None

    view_box = _attr("viewBox")

    def _num(val: str | None) -> float | None:
        if not val:
            return None
        try:
            # Remove a trailing 'px' if present
            return float(val.strip().removesuffix("px"))
        except Exception:
            return None

    width = _num(_attr("width"))
    height = _num(_attr("height"))
    return view_box, width, height, inner


def collate_svgs_as_grid(
    svgs: list[Path],
    labels: list[str],
    out_path: Path,
    rows: int | None,
    cols: int | None,
    cell_w: int,
    cell_h: int,
    pad: int,
    label_font_size: int,
    label_box_h: int,
    page_bg: str = "white",
) -> None:
    """
    Combine individual SVGs into a single SVG grid. Each tile inlines the molecule SVG
    (no external href). We annotate each cell with its index number in the upper-left.
    """
    n = len(svgs)
    # Determine effective rows/cols based on actual tile count (n).
    # If both rows and cols are provided, fit the content within those maxima
    # and shrink width/height to the necessary size for this page.
    if rows is not None and cols is not None:
        c = min(cols, n) if n > 0 else cols
        r = min(rows, math.ceil(n / c)) if c else rows
    else:
        r, c = infer_grid(n, rows, cols)

    width = c * cell_w + (c + 1) * pad
    height = r * cell_h + (r + 1) * pad

    # Build SVG header
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{page_bg}" />'
    )

    for idx, (svg_path, label) in enumerate(zip(svgs, labels)):
        row = idx // c
        col = idx % c
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)

        # Read and inline the child SVG
        svg_text = svg_path.read_text(encoding="utf-8")
        view_box, child_w, child_h, inner = _extract_svg_inner(svg_text)

        # Use full cell height for the image; numbering will overlay at top-left
        img_w = cell_w
        img_h = cell_h

        # Start tile group and background
        parts.append(f'<g transform="translate({x},{y})">')
        parts.append(
            f'  <rect x="0" y="0" width="{cell_w}" height="{cell_h}" fill="none" stroke="none" />'
        )

        # Embed the molecule SVG directly
        if view_box:
            parts.append(
                f'  <svg x="0" y="0" width="{img_w}" height="{img_h}" viewBox="{escape_xml(view_box)}">'
            )
            parts.append(inner)
            parts.append("  </svg>")
        elif child_w and child_h and child_w > 0 and child_h > 0:
            sx = img_w / child_w
            sy = img_h / child_h
            parts.append(f'  <g transform="scale({sx:.6f},{sy:.6f})">')
            parts.append(inner)
            parts.append("  </g>")
        else:
            parts.append(inner)

        # Number badge in upper-left
        badge_w = max(0, int(0.6 * label_font_size * max(1, len(str(label))) + 8))
        parts.append(
            f"  <g>"
            f'    <rect x="2" y="2" rx="4" ry="4" width="{badge_w}" height="{label_font_size + 6}" fill="white" opacity="0.85" />'
            f'    <text x="6" y="{2 + label_font_size}" font-size="{label_font_size}" font-family="Helvetica, Arial, sans-serif" text-anchor="start" fill="black">{escape_xml(str(label))}</text>'
            f"  </g>"
        )
        parts.append("</g>")

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render SMILES to SVGs (Open Babel) and collate into a labeled SVG grid."
    )
    p.add_argument(
        "input",
        type=Path,
        help="Text file with one SMILES per line (comments with '#').",
    )
    p.add_argument(
        "grid_out",
        type=Path,
        help="Output path for the collated grid SVG.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write individual SVGs.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help=(
            "Number of rows in the grid. If both rows and cols are provided "
            "and there are more molecules than rows*cols, output is split into multiple pages."
        ),
    )
    p.add_argument(
        "--cols",
        type=int,
        default=None,
        help=(
            "Number of columns in the grid. If both rows and cols are provided "
            "and there are more molecules than rows*cols, output is split into multiple pages."
        ),
    )
    p.add_argument("--cell-w", type=int, default=256, help="Cell width in pixels.")
    p.add_argument(
        "--cell-h",
        type=int,
        default=200,
        help="Cell height in pixels (includes label area).",
    )
    p.add_argument(
        "--pad", type=int, default=16, help="Padding between cells in pixels."
    )
    p.add_argument(
        "--label-font-size", type=int, default=12, help="Font size for SMILES labels."
    )
    p.add_argument(
        "--label-box-h",
        type=int,
        default=28,
        help="Reserved height for the label beneath each image.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="mol",
        help="Filename prefix for per-molecule SVGs.",
    )
    p.add_argument(
        "--obabel",
        type=str,
        default="obabel",
        help="Path to the Open Babel CLI executable.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if shutil.which(args.obabel) is None:
        print(
            f"ERROR: Open Babel CLI '{args.obabel}' not found on PATH. Install Open Babel and try again.",
            file=sys.stderr,
        )
        return 2

    if args.input:
        smiles_list = load_smiles_from_file(args.input)
    else:
        smiles_list = [s.strip() for s in args.smiles or [] if s.strip()]

    if not smiles_list:
        print("No SMILES provided.", file=sys.stderr)
        return 1

    # Deduplicate while preserving order (so labels match grid order)
    seen: set[str] = set()
    ordered_smiles: list[str] = []
    for s in smiles_list:
        if s not in seen:
            seen.add(s)
            ordered_smiles.append(s)

    tmpdir = None
    if args.out_dir is not None:
        out_dir: Path = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmpdir = TemporaryDirectory()
        out_dir = Path(tmpdir.name)

    svg_paths: list[Path] = []
    for i, smi in enumerate(ordered_smiles, start=1):
        svg_path = out_dir / f"{args.prefix}_{i:03d}.svg"
        print(f"{smi}: {svg_path.name}")
        run_obabel_to_svg(smi, svg_path, obabel=args.obabel)
        svg_paths.append(svg_path)

    # Collate: if both rows and cols are specified and there are more tiles
    # than rows*cols, split across multiple SVG pages using a numbered suffix.
    total = len(svg_paths)
    wrote_pages: list[Path] = []
    if args.rows is not None and args.cols is not None:
        per_page = args.rows * args.cols
        num_pages = math.ceil(total / per_page) if per_page > 0 else 1
    else:
        per_page = total
        num_pages = 1

    def page_out_path(base: Path, page_idx: int, total_pages: int) -> Path:
        if total_pages <= 1:
            return base
        stem = base.stem
        suffix = base.suffix
        return base.with_name(f"{stem}_p{page_idx + 1:02d}{suffix}")

    for page_idx in range(num_pages):
        start = page_idx * per_page
        end = min(total, (page_idx + 1) * per_page)
        page_svgs = svg_paths[start:end]
        page_labels = [str(i) for i in range(start + 1, end + 1)]
        out_path = page_out_path(args.grid_out, page_idx, num_pages)
        collate_svgs_as_grid(
            svgs=page_svgs,
            labels=page_labels,
            out_path=out_path,
            rows=args.rows,
            cols=args.cols,
            cell_w=args.cell_w,
            cell_h=args.cell_h,
            pad=args.pad,
            label_font_size=args.label_font_size,
            label_box_h=args.label_box_h,
        )
        wrote_pages.append(out_path)

    print(f"Wrote {len(svg_paths)} molecule SVGs to: {out_dir}")
    if len(wrote_pages) == 1:
        print(f"Wrote grid SVG to: {wrote_pages[0]}")
    else:
        print("Wrote grid SVG pages:")
        for p in wrote_pages:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
