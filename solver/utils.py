import tikzplotlib
import re
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt


def save_tikz(filename):
    if not re.match(r'.*\.tikz$', filename):
        filename = f'{filename}.tikz'
    tikzplotlib.clean_figure()
    tikzplotlib.save(filename)
