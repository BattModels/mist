#!/usr/bin/env python
"""Plots over `token,count` CSVs written by token_counts.py."""

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.lines import Line2D

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e5e4e0",
        "guide": "#a8a7a1",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "guide": "#6b6a64",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

cli = typer.Typer()


def load(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    assert {"token", "count"} <= set(df.columns), f"{path} needs token,count columns"
    df["token"] = df["token"].astype(str)  # keep '0'..'9' and 'null' as text
    return df.groupby("token")["count"].sum()


def themes(theme: str) -> List[str]:
    return ["light", "dark"] if theme == "both" else [theme]


def themed_path(out: Path, theme: str) -> Path:
    if theme == "light":
        return out
    return out.with_name(f"{out.stem}_dark{out.suffix}")


def style(ax, t, axis="y", boxed=False):
    ax.set_facecolor(t["surface"])
    ax.grid(True, axis=axis, color=t["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    sides = ("top", "right", "left", "bottom") if boxed else ("left", "bottom")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in sides)
        ax.spines[side].set_color(t["muted"] if boxed else t["grid"])
        ax.spines[side].set_linewidth(0.9)
    for which, length, width in (("major", 4, 0.9), ("minor", 2.5, 0.7)):
        ax.tick_params(
            which=which,
            colors=t["muted"],
            labelsize=9,
            direction="in",
            top=False,
            right=False,
            length=length,
            width=width if boxed else 0,
        )


@cli.command()
def histogram(
    csvs: List[Path],
    labels: Optional[List[str]] = None,
    output: Path = Path("token_histograms.png"),
    log: bool = False,
    share: bool = False,
    theme: str = "light",
):
    """One bar histogram per dataset, stacked on a shared token axis."""
    labels = labels or [c.stem for c in csvs]
    assert len(labels) == len(csvs), "need one label per CSV"
    series = [load(c) for c in csvs]

    normed = [100 * s / s.sum() for s in series]
    values = normed if share else series
    order = (
        pd.concat(normed, axis=1)
        .fillna(0.0)
        .sum(axis=1)
        .sort_values(ascending=False)
        .index
    )
    x = np.arange(len(order))
    unit = "share of tokens (%)" if share else "occurrences"

    for name in themes(theme):
        t = THEMES[name]
        fig, axes = plt.subplots(
            len(series),
            1,
            sharex=True,
            squeeze=False,
            figsize=(max(7.0, 0.3 * len(order)), 2.7 * len(series)),
        )
        fig.patch.set_facecolor(t["surface"])

        for i, (v, label) in enumerate(zip(values, labels)):
            ax = axes[i, 0]
            vals = v.reindex(order).fillna(0).values
            ax.bar(
                x, vals, width=0.72, color=t["series"][i % len(t["series"])], zorder=3
            )
            if log:
                ax.set_yscale("log")
                pos = vals[vals > 0]
                if len(pos):
                    ax.set_ylim(bottom=pos.min() / 2)
            elif not share:
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:,.0f}")
                )
            ax.set_ylabel(f"{label}\n{unit}", color=t["ink"], fontsize=10)
            style(ax, t, boxed=True)

        ax = axes[-1, 0]
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontfamily="monospace", fontsize=9)
        ax.set_xlim(-0.8, len(order) - 0.2)
        ax.set_xlabel("Token", color=t["ink"], fontsize=10)

        fig.tight_layout()
        path = themed_path(output, name)
        fig.savefig(path, dpi=200, facecolor=t["surface"])
        plt.close(fig)
        print("wrote", path)


@cli.command()
def compare(
    csv_a: Path,
    csv_b: Path,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    output: Path = Path("token_comparison.png"),
    top: Optional[int] = None,
    theme: str = "light",
):
    """Share-of-tokens dumbbell plus log2 enrichment of A over B."""
    label_a = label_a or csv_a.stem
    label_b = label_b or csv_b.stem
    a, b = load(csv_a), load(csv_b)

    df = pd.DataFrame({"a": 100 * a / a.sum(), "b": 100 * b / b.sum()}).fillna(0.0)
    df = df.sort_values("b", ascending=False)
    if top:
        df = df.head(top)
    both = (df.a > 0) & (df.b > 0)
    df["enrich"] = np.where(both, np.log2(df.a.where(both) / df.b.where(both)), np.nan)
    df.to_csv(output.with_suffix(".csv"))

    only_a = sorted(df.index[df.b == 0])
    only_b = sorted(df.index[df.a == 0])
    print(f"only in {label_a}: {only_a}\nonly in {label_b}: {only_b}")

    n = len(df)
    y = np.arange(n)[::-1]
    for name in themes(theme):
        t = THEMES[name]
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(11.4, 1.9 + 0.235 * n),
            gridspec_kw={"width_ratios": [2.15, 1], "wspace": 0.06},
        )
        fig.patch.set_facecolor(t["surface"])

        floor = min(df.loc[df.a > 0, "a"].min(), df.loc[df.b > 0, "b"].min()) / 2.5
        for yi, (_, r) in zip(y, df.iterrows()):
            ax1.plot(
                [r.a or floor, r.b or floor],
                [yi, yi],
                color=t["guide"],
                lw=1.0,
                alpha=0.7,
                zorder=2,
            )
            if r.a == 0 or r.b == 0:
                ax1.plot(
                    floor,
                    yi,
                    marker="x",
                    ms=6,
                    mew=1.4,
                    zorder=4,
                    color=t["series"][0 if r.a == 0 else 1],
                )
        for col, slot in (("a", 0), ("b", 1)):
            present = df[col] > 0
            ax1.scatter(
                df[col][present],
                y[present.values],
                s=46,
                zorder=3,
                c=t["series"][slot],
                linewidths=0.6,
                edgecolors=t["surface"],
            )
        ax1.set_xscale("log")
        ax1.set_xlim(floor / 1.8, 100)
        ax1.set_yticks(y)
        ax1.set_yticklabels(df.index, fontfamily="monospace", fontsize=9)
        ax1.set_xlabel("Share of all tokens (%)", color=t["ink"], fontsize=9)

        vals = np.nan_to_num(df.enrich.values)
        ax2.barh(
            y,
            vals,
            height=0.62,
            zorder=3,
            color=[t["series"][0 if v > 0 else 1] for v in vals],
        )
        ax2.axvline(0, color=t["guide"], lw=1.0, zorder=2)
        ax2.set_xlim(
            -max(3.0, np.abs(vals).max() * 1.15), max(3.0, np.abs(vals).max() * 1.15)
        )
        ax2.set_yticks(y)
        ax2.set_yticklabels([])
        ax2.set_xlabel(
            f"log₂ enrichment ({label_a} / {label_b})", color=t["ink"], fontsize=9
        )

        for ax in (ax1, ax2):
            ax.set_ylim(-0.8, n - 0.2)
            style(ax, t, axis="x")

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=7,
                label=lab,
                markerfacecolor=t["series"][i],
                markeredgecolor=t["surface"],
            )
            for i, lab in enumerate((label_a, label_b))
        ]
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                ls="",
                markersize=6,
                markeredgewidth=1.4,
                color=t["muted"],
                label="absent from that corpus",
            )
        )
        leg = fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.06 / fig.get_figheight()),
        )
        for txt in leg.get_texts():
            txt.set_color(t["muted"])

        fig.tight_layout(rect=[0, 0.42 / fig.get_figheight(), 1, 1])
        path = themed_path(output, name)
        fig.savefig(path, dpi=200, facecolor=t["surface"])
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    cli()
