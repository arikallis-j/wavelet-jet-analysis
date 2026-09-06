import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial
from .plots import PLOTS, PLOTS_COLORBAR

FIELDS = {
    'q': 'q',
    'v': 'v',
    'p': 'p',
    'e': 'eh',
    'E': 'Eh',
}

def draw_frame(t_idx, fig, axes, ds, fields=['q', 'v', 'p', 'e'], kwargs={}, colorbar=False, progress=False):
    if progress:
        print(f"iter: {t_idx+1}/{len(ds.time)}")
    artists = []
    fig.suptitle(f"Time t = {ds.time[t_idx]:.1f}")
    for ax, name in zip(axes.flat, fields):
        ax.clear()
        artist = PLOTS[name](ds, ax, t_idx, **kwargs.get(name, {}))
        if colorbar and PLOTS_COLORBAR[name]:
            fig.colorbar(artist, ax=ax, shrink=0.8, format='{x:+.1f}')
        artists.append(artist)
    fig.tight_layout()
    return artists

def draw_snapshot(ds, t, path = None, show=True, fields=['q', 'v', 'p', 'e'], kwargs={}, colorbar=False, size=5):
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields, figsize=(size*n_fields, size))
    if n_fields == 1:
        axes = np.array(axes)
    draw_frame(t, fig, axes, ds, fields=fields, kwargs=kwargs, colorbar=colorbar)
    if path is not None:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()

def draw_animation(ds, path = None, show=True, fps=10, fields=['q', 'v', 'p'], kwargs={}, colorbar=False, size=5, progress = False):
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields, figsize=(size*n_fields, size))
    if n_fields == 1:
        axes = np.array(axes)
    draw_frame(0, fig, axes, ds, fields, kwargs, colorbar=colorbar)
    update = partial(draw_frame, fig=fig, axes=axes, ds=ds, fields=fields, progress=progress)
    ani = animation.FuncAnimation(fig, update, frames=len(ds.time), interval=100, blit=False)
    if path is not None:
        ani.save(path, writer='ffmpeg', fps=fps, dpi=150, bitrate=1800)
    if show:
        plt.show()
    plt.close()