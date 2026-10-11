"""One style for every figure of the documentation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402,F401

OUT = Path(__file__).resolve().parents[1] / "img"

# print widths, inches
FULL = 7.2
HALF = 3.5

MIN_PT = 7.0
BASE_PT = 8.0
DPI = 300

# the journals' sans-serif where installed, else the one matplotlib ships
_FONTS = {f.name for f in font_manager.fontManager.ttflist}
SANS = next(f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if f in _FONTS)

# the Okabe-Ito palette, colour-blind safe
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"
LIGHT = "#E8E8E8"
PAPER = "#FFFFFF"

ICE = "#DCEBF5"
WATER = "#BFDDF0"
BED = "#C9B79C"

SCENARIO = {"historical": BLACK, "ssp126": BLUE, "ssp585": VERMILION}
RUN_STYLE = {"control": dict(ls="--", lw=1.0), "ocean": dict(ls="-", lw=1.4)}
HOST = {"OGGM": BLUE, "PyGEM": ORANGE, "GloGEM": GREEN}

plt.rcParams.update({
    "font.family": SANS,
    "font.size": BASE_PT,
    "axes.titlesize": BASE_PT + 1,
    "axes.labelsize": BASE_PT,
    "xtick.labelsize": MIN_PT,
    "ytick.labelsize": MIN_PT,
    "legend.fontsize": MIN_PT,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": BLACK,
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "legend.frameon": False,
    # an opaque page, so the figure reads the same on a light and a dark site
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.transparent": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": SANS,
    "mathtext.cal": SANS,
    "mathtext.sf": SANS,
    "mathtext.it": f"{SANS}:italic",
    "mathtext.bf": f"{SANS}:bold",
    "mathtext.fallback": "stixsans",
    "svg.fonttype": "path",
    "svg.hashsalt": "tidewater",
})


def check_text(fig):
    """Raise if any visible text of ``fig`` is smaller than the floor."""
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(plt.Text)
             if t.get_visible() and t.get_text().strip()
             and t.get_fontsize() < MIN_PT - 1e-9]
    if small:
        raise ValueError(f"text below {MIN_PT} pt: {small[:5]}")


def save(fig, name, out=None):
    """Write ``name``.svg and ``name``.png after the size check."""
    check_text(fig)
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", dpi=DPI, metadata={"Date": None, "Creator": None})
    fig.savefig(out / f"{name}.png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return out / f"{name}.svg"


def box(ax, x, y, w, h, title, body="", color=BLUE, fill=None, title_size=BASE_PT,
        body_size=MIN_PT, align="center", foot=""):
    """A rounded box with a bold title, a body and a grey foot, in axes data units."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                           fc=fill or PAPER, ec=color, lw=1.0, mutation_aspect=1)
    ax.add_patch(patch)
    tx, dx = (x + w / 2, 0) if align == "center" else (x, 5)
    ax.annotate(title, (tx, y + h), xytext=(dx, -4), textcoords="offset points",
                ha=align, va="top", fontsize=title_size, fontweight="bold",
                color=BLACK)
    if body:
        drop = 4 + 1.3 * title_size * (1 + title.count("\n")) + 2
        ax.annotate(body, (tx, y + h), xytext=(dx, -drop), textcoords="offset points",
                    ha=align, va="top", fontsize=body_size, color=BLACK,
                    linespacing=1.35)
    if foot:
        ax.annotate(foot, (tx, y), xytext=(dx, 4), textcoords="offset points",
                    ha=align, va="bottom", fontsize=MIN_PT, color=GREY)
    return patch


def arrow(ax, xy0, xy1, color=GREY, lw=1.0, style="-|>", text=None, text_xy=None,
          **text_kw):
    """An arrow between two points in data units, with an optional label."""
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=8,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))
    if text:
        tx, ty = text_xy or ((xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2)
        ax.text(tx, ty, text, fontsize=text_kw.pop("fontsize", MIN_PT),
                color=text_kw.pop("color", BLACK), **text_kw)


def canvas(width, height):
    """A figure with one blank axes spanning [0, 1] x [0, 1]."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def stamp(fig, text, color=VERMILION):
    """A label in the upper right corner of a figure, e.g. that its data are synthetic."""
    fig.text(0.995, 0.995, text, ha="right", va="top", fontsize=MIN_PT,
             fontweight="bold", color=color)
