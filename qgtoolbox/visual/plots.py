import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo

def plot_potential_vorticity(ds, axis, t_idx, cmap='cmo.curl', levels=50):
    x = ds.x.values
    y = ds.y.values
    q = ds.q.isel(time=t_idx).values.T
    Q = np.abs(ds.q).max().values
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, q/Q, cmap=cmap, levels=scale)
    axis.set_title(f'Potential Vorticity $q$ | $Q = {Q:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_absolute_velocity(ds, axis, t_idx, cmap='cmo.speed', levels=50):
    x = ds.x.values
    y = ds.y.values
    v = ds.v.isel(time=t_idx).values.T
    V = np.abs(ds.v).max().values
    scale = np.linspace(0.0, +1, levels)
    cf = axis.contourf(x, y, v/V, cmap=cmap, levels=scale)
    axis.set_title(f'Absolute Velocity $v$, | $V = {V:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

def plot_stream_function(ds, axis, t_idx, cmap='cmo.delta', levels=50):
    x = ds.x.values
    y = ds.y.values
    p = ds.p.isel(time=t_idx).values.T
    P = np.abs(ds.p).max().values
    scale = np.linspace(-1, +1, levels)
    cf = axis.contourf(x, y, p/P, cmap=cmap, levels=scale)
    axis.set_title(f'Stream Function $\\psi$ | $\\Psi = {P:.1e}$')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    return cf

PLOTS = {
    'q': plot_potential_vorticity,
    'v': plot_absolute_velocity,
    'p': plot_stream_function,
}