import numpy as np
import xrft

def compute_abs_velocity(ds):
    v = np.sqrt(ds['ux']**2 + ds['uy']**2)
    v.attrs['long_name'] = 'absolute velocity anomaly'
    v.attrs['units'] = 'm s^-1'
    return v

def compute_mean_energy(ds):
    e = 0.5 * (ds.ux**2 + ds.uy**2)
    e_mean = e.mean(dim=['x', 'y'])
    e_mean.attrs['long_name'] = 'mean kinetic energy'
    e_mean.attrs['units'] = 'm^2 s^-2'
    return e_mean

def compute_2d_spectrum(ds, dims=['y', 'x']):
    uxh = xrft.fft(ds.ux, dim=dims)
    uyh = xrft.fft(ds.uy, dim=dims)
    eh = np.abs(uxh)**2 + np.abs(uyh)**2
    eh = eh.assign_coords({
        'freq_x': 2 * np.pi * eh.freq_x,
        'freq_y': 2 * np.pi * eh.freq_y
    }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
    eh.attrs['long_name'] = '2d energy spectrum'
    eh.attrs['units'] = 'm^2 s^-2'
    return eh

def compute_1d_spectrum(ds, fftdim=['kx', 'ky'], nfactor=4):
    Eh = xrft.isotropize(ds.eh, fftdim=fftdim, nfactor=nfactor)
    Eh = Eh.rename({'freq_r': 'k'})
    Eh.attrs['long_name'] = '1d energy spectrum'
    Eh.attrs['units'] = 'm^2 s^-2'
    return Eh

def calc_diganostic(ds):
    ds['v'] = compute_abs_velocity(ds)
    ds['e_mean'] = compute_mean_energy(ds)
    ds['eh'] = compute_2d_spectrum(ds)
    ds['Eh'] = compute_1d_spectrum(ds)
    return ds