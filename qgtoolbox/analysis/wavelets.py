import numpy as np
import xarray as xr
import xwavelet.wavelet as xwl
import xrft

def compute_cwt_spectrum(ds, dims=['y', 'x'], wtype='imorlet', gamma_idx=0, direction='x', n_scale=None):
    n_grid = ds.attrs['grid.n_grid']
    x0 = ds.x.values[1] - ds.x.values[0]
    if n_scale is None:
        n_scale = int(np.floor(np.log(n_grid)/np.log(2)))
    s_vals = np.geomspace(2.0, n_grid, n_scale)
    scale = xr.DataArray(s_vals, dims=["s"], coords={"s": s_vals})
    
    Wux = xwl.cwt2_field(ds.ux, scale, dim=['y', 'x'], x0=x0, wtype=wtype, n_gamma=n_grid, gamma_idx=gamma_idx, direction=direction, n_theta=4)
    Wuy = xwl.cwt2_field(ds.uy, scale, dim=['y', 'x'], x0=x0,  wtype=wtype, n_gamma=n_grid, gamma_idx=gamma_idx, direction=direction, n_theta=4)
    Wex = np.abs(Wux)**2 / Wux["s"] * x0**2
    Wey = np.abs(Wuy)**2 / Wuy["s"] * x0**2
    We = Wex + Wey
    
    We.attrs['long_name'] = 'energy wavelet spectrum'
    We.attrs['units'] = 'm^2 s^-2'
    return We


def compute_swt_radial_spectrum(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cH']**2 + uxw['cV']**2 + uxw['cD']**2
    eyw = uyw['cH']**2 + uyw['cV']**2 + uyw['cD']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = 'energy wavelet spectrum'
    ew.attrs['units'] = 'm^2 s^-2'
    return ew
    
def compute_swt_horizontal_spectrum_field(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cH']**2
    eyw = uyw['cH']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = 'horizontal energy wavelet spectrum'
    ew.attrs['units'] = 'm^2 s^-2'
    return ew

def compute_swt_vertical_spectrum_field(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cV']**2
    eyw = uyw['cV']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = 'vertical energy wavelet spectrum'
    ew.attrs['units'] = 'm^2 s^-2'
    return ew

def compute_zonal_cwt_spectrum(ds):
    Ewh = ds.ewh.mean(dim='y').isel(x=0)
    Ewh.attrs['long_name'] = 'zonal energy wavelet spectrum'
    Ewh.attrs['units'] = 'm^2 s^-2'
    return Ewh

def compute_meridional_cwt_spectrum(ds):
    Ewv = ds.ewv.mean(dim='x').isel(y=0)
    Ewv.attrs['long_name'] = 'meridional energy wavelet spectrum'
    Ewv.attrs['units'] = 'm^2 s^-2'
    return Ewv


def calc_wavelets(ds, wtype='imporlet', gamma_idx=0, n_scale=None):
    # ds.attrs['grid.n_lev'] =  xwl.swt_max_level(ds.attrs['grid.n_grid'])
    # ds['ew'] = compute_swt_radial_spectrum(ds, wtype=wtype)
    # ds['ewh'] = compute_swt_horizontal_spectrum(ds, wtype=wtype)
    # ds['ewv'] = compute_swt_vertical_spectrum(ds, wtype=wtype)
    ds['ewh'] = compute_cwt_spectrum(ds, wtype=wtype, gamma_idx=gamma_idx, direction='x', n_scale=n_scale)
    ds['ewv'] = compute_cwt_spectrum(ds, wtype=wtype, gamma_idx=gamma_idx, direction='y', n_scale=n_scale)
    if 'angle' in ds['ewv'].dims:
        ds['ewv'] = ds['ewv'].isel(angle=1)
        ds['ewh'] = ds['ewh'].isel(angle=0)
    ds['Ewh'] = compute_zonal_cwt_spectrum(ds)
    ds['Ewv'] = compute_meridional_cwt_spectrum(ds)
    return ds