import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial
from .plots import PLOTS, PLOTS_COLORBAR, FIELDS


def draw_frame(t_idx, fig, axes, ds, fields=['q', 'v', 'p', 'e'], kwargs={}, colorbar=False, progress=False):
    if progress:
        print(f"iter: {t_idx+1}/{len(ds.time)}")
    artists = []
    kwargs['ew']['swt'] = 0
    fig.suptitle(f"Time t = {ds.time[t_idx]:.1f}")
    for ax, name in zip(axes.flat, fields):
        ax.clear()
        artist = PLOTS[name](ds, ax, t_idx, **kwargs.get(name, {}))
        if colorbar and PLOTS_COLORBAR[name]:
            fig.colorbar(artist, ax=ax, shrink=0.8, format='{x:+.1f}')
        artists.append(artist)
        if name == 'ew':
            kwargs['ew']['swt'] += 1
    
    return artists

def draw_snapshot(ds, t, path = None, show=True, fields=['q', 'v', 'p', 'e'], kwargs={}, colorbar=False, size=5):
    n_fields = len(fields)
    if n_fields%3 == 0 and n_fields!=3:
        fig, axes = plt.subplots(3, n_fields//3, figsize=(size*n_fields//3, size*3))
    elif n_fields%2 == 0 and n_fields!=2:
        fig, axes = plt.subplots(2, n_fields//2, figsize=(size*n_fields//2, size*2))
    else:
       fig, axes = plt.subplots(1, n_fields, figsize=(size*n_fields, size))
        
    if n_fields == 1:
        axes = np.array(axes)
    draw_frame(t, fig, axes, ds, fields=fields, kwargs=kwargs, colorbar=colorbar)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()

def draw_animation(ds, path = None, show=True, fps=10, fields=['q', 'v', 'p'], kwargs={}, colorbar=False, size=5, progress = False):
    n_fields = len(fields)
    if n_fields%3 == 0 and n_fields!=3:
        fig, axes = plt.subplots(3, n_fields//3, figsize=(size*n_fields//3, size*3))
    elif n_fields%2 == 0 and n_fields!=2:
        fig, axes = plt.subplots(2, n_fields//2, figsize=(size*n_fields//2, size*2))
    else:
       fig, axes = plt.subplots(1, n_fields, figsize=(size*n_fields, size))
        
    if n_fields == 1:
        axes = np.array(axes)
    draw_frame(0, fig, axes, ds, fields, kwargs, colorbar=colorbar)
    fig.tight_layout()
    update = partial(draw_frame, fig=fig, axes=axes, ds=ds, fields=fields, kwargs=kwargs, progress=progress)
    ani = animation.FuncAnimation(fig, update, frames=len(ds.time), interval=100, blit=False)
    if path is not None:
        ani.save(path, writer='ffmpeg', fps=fps, dpi=150)
    if show:
        plt.show()
    plt.close()