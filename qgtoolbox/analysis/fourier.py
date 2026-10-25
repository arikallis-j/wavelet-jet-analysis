import numpy as np
import xarray as xr
import xrft

def compute_spectrum_field(ds, dims=['y', 'x']):
    uxh = xrft.fft(ds.ux, dim=dims)
    uyh = xrft.fft(ds.uy, dim=dims)
    eh = np.abs(uxh)**2 + np.abs(uyh)**2
    eh = eh.assign_coords({
        'freq_x': 2 * np.pi * eh.freq_x,
        'freq_y': 2 * np.pi * eh.freq_y,
    }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
    eh.attrs['long_name'] = 'energy spectrum'
    eh.attrs['units'] = 'm^2 s^-2'
    return eh

def compute_zonal_spectrum_field(ds):
    n_grid = ds.attrs['grid.n_grid']
    uxhx = xrft.fft(ds.ux, dim=['x'])
    uyhx = xrft.fft(ds.uy, dim=['x'])
    ehx = np.abs(uxhx)**2 + np.abs(uyhx)**2
    ehx = ehx.assign_coords({
        'freq_x': 2 * np.pi * ehx.freq_x,
    }).rename({'freq_x': 'kx'})
    ehx = 4 * np.pi**2/n_grid * ehx
    ehx.attrs['long_name'] = 'zonal energy spectrum'
    ehx.attrs['units'] = 'm^2 s^-2'
    return ehx

def compute_meridional_spectrum_field(ds):
    n_grid = ds.attrs['grid.n_grid']
    uxhy = xrft.fft(ds.ux, dim=['y'])
    uyhy = xrft.fft(ds.uy, dim=['y'])
    ehy = np.abs(uxhy)**2 + np.abs(uyhy)**2
    ehy = ehy.assign_coords({
        'freq_y': 2 * np.pi * ehy.freq_y,
    }).rename({'freq_y': 'ky'})
    ehy = 4 * np.pi**2/n_grid  * ehy
    ehy.attrs['long_name'] = 'meridional energy spectrum'
    ehy.attrs['units'] = 'm^2 s^-2'
    return ehy

def compute_radial_spectrum(ds, fftdim=['kx', 'ky'], nfactor=4):
    Eh = xrft.isotropize(ds.eh, fftdim=fftdim, nfactor=nfactor)
    Eh = Eh.rename({'freq_r': 'k'})
    Eh.attrs['long_name'] = 'radial energy spectrum'
    Eh.attrs['units'] = 'm^2 s^-2'
    return Eh

def compute_zonal_spectrum(ds):
    Ehx = ds.eh.mean(dim='ky')
    Ehx.attrs['long_name'] = 'zonal energy spectrum'
    Ehx.attrs['units'] = 'm^2 s^-2'
    return Ehx

def compute_meridional_spectrum(ds):
    Ehy = ds.eh.mean(dim='kx')
    Ehy.attrs['long_name'] = 'meridional energy spectrum'
    Ehy.attrs['units'] = 'm^2 s^-2'
    return Ehy

def calc_fourier(ds):
    ds['eh'] = compute_spectrum_field(ds)
    ds['ehx'] = compute_zonal_spectrum_field(ds)
    ds['ehy'] = compute_meridional_spectrum_field(ds)
    ds['Ehr'] = compute_radial_spectrum(ds)
    ds['Ehx'] = compute_zonal_spectrum(ds)
    ds['Ehy'] = compute_meridional_spectrum(ds)
    return ds