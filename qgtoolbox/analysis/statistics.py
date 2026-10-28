import numpy as np

def compute_potential_vorticity(ds):
    qp = ds.q + ds.attrs['model.beta'] * ds.y
    qp.attrs['long_name'] = 'potential vorticity'
    qp.attrs['units'] = 's^-1'
    return qp

def compute_absolute_velocity(ds):
    av = np.sqrt(ds.ux**2 + ds.uy**2)
    av.attrs['long_name'] = 'absolute velocity'
    av.attrs['units'] = 'm s^-1'
    return av

def compute_energy_density(ds):
    e = 0.5 * (ds.ux**2 + ds.uy**2)
    e.attrs['long_name'] = 'energy density'
    e.attrs['units'] = 'm^2 s^-2 / m^2'
    return e

def compute_enstrophy_density(ds):
    z = 0.5 * (ds.q**2)
    z.attrs['long_name'] = 'enstrophy density'
    z.attrs['units'] = 's^-2 / m^2'
    return z

def calc_statistics(ds):
    ds['qp'] = compute_potential_vorticity(ds)
    ds['v']  = compute_absolute_velocity(ds)
    ds['e']  = compute_energy_density(ds)
    ds['z']  = compute_enstrophy_density(ds)
    return ds