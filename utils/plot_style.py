"""NeurIPS-conformant matplotlib style + shared palette."""

from pathlib import Path

import matplotlib.pyplot as plt


NEURIPS_TEXT_WIDTH_IN = 5.5

PRIMARY   = "#0072B2"
SECONDARY = "#D55E00"
REFERENCE = "#7F7F7F"
THRESHOLD = "#E69F00"

POINT_KW = dict(
    markersize=4.5,
    markeredgecolor="black",
    markeredgewidth=0.5,
    capsize=3,
    elinewidth=1.0,
    linewidth=0,
)


def apply_neurips_style():
    """Set rcParams for NeurIPS-style figures (10pt serif, TrueType fonts)."""
    plt.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.titlesize":     10,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "lines.linewidth":    1.2,
        "lines.markersize":   4.5,
        "legend.frameon":     True,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.6",
        "figure.facecolor":   "white",
    })


def savefig_neurips(fig, stem, outdir):
    """Save figure as PDF (for LaTeX) and PNG (for previewing)."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
