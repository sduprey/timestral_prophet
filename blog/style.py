"""Shared look for the blog figures."""
import matplotlib.pyplot as plt

TIMESTRAL, PROPHET, OTHER = "#2a78d6", "#eb6834", "#b4b2a9"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": BG,
    "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})
