import numpy as np
import xarray as xr
import xrft
import xwavelet.wavelet as xwl
from tqdm import tqdm

from scipy.signal import find_peaks

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

def compute_x_vorticity(ds, dims=['y']):
    qx = ds.q.mean(dim=dims)
    qx.attrs['long_name'] = 'vorticity along x'
    qx.attrs['units'] = 's^-1'
    return qx

def compute_y_vorticity(ds, dims=['x']):
    qy = ds.q.mean(dim=dims)
    qy.attrs['long_name'] = 'vorticity along y'
    qy.attrs['units'] = 's^-1'
    return qy

def compute_mean_y_velocity(ds, dims=['y']):
    vy = ds.uy.mean(dim=dims)
    vy.attrs['long_name'] = 'velocity along y'
    vy.attrs['units'] = 's^-1'
    return vy

def compute_mean_x_velocity(ds, dims=['x']):
    vx = ds.ux.mean(dim=dims)
    vx.attrs['long_name'] = 'velocity along x'
    vx.attrs['units'] = 's^-1'
    return vx

def compute_2d_spectrum(ds, dims=['y', 'x']):
    uxh = xrft.fft(ds.ux, dim=dims)
    uyh = xrft.fft(ds.uy, dim=dims)
    eh = np.abs(uxh)**2 + np.abs(uyh)**2
    eh = eh.assign_coords({
        'freq_x': 2 * np.pi * eh.freq_x,
        'freq_y': 2 * np.pi * eh.freq_y,
    }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
    eh.attrs['long_name'] = '2d energy spectrum'
    eh.attrs['units'] = 'm^2 s^-2'
    return eh

def compute_2d_spectrum_x(ds):
    uxhx = xrft.fft(ds.ux, dim=['x'])
    uyhx = xrft.fft(ds.uy, dim=['x'])
    ehx = np.abs(uxhx)**2 + np.abs(uyhx)**2
    ehx = ehx.assign_coords({
        'freq_x': 2 * np.pi * ehx.freq_x,
    }).rename({'freq_x': 'kx'})
    # Ehx = ehx.mean(dim='y')
    ehx.attrs['long_name'] = '2d energy spectrum along x'
    ehx.attrs['units'] = 'm^2 s^-2'
    return ehx

def compute_1d_spectrum(ds, fftdim=['kx', 'ky'], nfactor=4):
    Eh = xrft.isotropize(ds.eh, fftdim=fftdim, nfactor=nfactor)
    Eh = Eh.rename({'freq_r': 'k'})
    Eh.attrs['long_name'] = '1d energy spectrum'
    Eh.attrs['units'] = 'm^2 s^-2'
    return Eh

# def compute_1d_spectrum_x(ds, dims=['ky']):
#     uxh = xrft.fft(ds.ux, dim=['y', 'x'])
#     uyh = xrft.fft(ds.uy, dim=['y', 'x'])
#     eh = np.abs(uxh)**2 + np.abs(uyh)**2
#     eh = eh.assign_coords({
#         'freq_x': 2 * np.pi * eh.freq_x,
#         'freq_y': 2 * np.pi * eh.freq_y,
#     }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
#     Ehx = eh.mean(dim=dims)
#     Ehx.attrs['long_name'] = '1d energy spectrum along x'
#     Ehx.attrs['units'] = 'm^2 s^-2'
#     return Ehx

def compute_1d_spectrum_x(ds):
    n_grid = ds.attrs['grid.n_grid']
    uxhx = xrft.fft(ds.ux, dim=['x'])
    uyhx = xrft.fft(ds.uy, dim=['x'])
    ehx = np.abs(uxhx)**2 + np.abs(uyhx)**2
    ehx = ehx.assign_coords({
        'freq_x': 2 * np.pi * ehx.freq_x,
    }).rename({'freq_x': 'kx'})
    Ehx = ehx.mean(dim='y')
    Ehx.attrs['long_name'] = '1d energy spectrum along x'
    Ehx.attrs['units'] = 'm^2 s^-2'
    return Ehx


# def compute_1d_spectrum_y(ds, dims=['kx']):
#     uxh = xrft.fft(ds.ux, dim=['y', 'x'])
#     uyh = xrft.fft(ds.uy, dim=['y', 'x'])
#     eh = np.abs(uxh)**2 + np.abs(uyh)**2
#     eh = eh.assign_coords({
#         'freq_x': 2 * np.pi * eh.freq_x,
#         'freq_y': 2 * np.pi * eh.freq_y,
#     }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
#     Ehy = eh.mean(dim=dims)
#     Ehy.attrs['long_name'] = '1d energy spectrum along y'
#     Ehy.attrs['units'] = 'm^2 s^-2'
#     return Ehy

def compute_1d_spectrum_y(ds):
    n_grid = ds.attrs['grid.n_grid']
    uxhy = xrft.fft(ds.ux, dim=['y'])
    uyhy = xrft.fft(ds.uy, dim=['y'])
    ehy = np.abs(uxhy)**2 + np.abs(uyhy)**2
    ehy = ehy.assign_coords({
        'freq_y': 2 * np.pi * ehy.freq_y,
    }).rename({'freq_y': 'ky'})
    Ehy = ehy.mean(dim='x')
    Ehy.attrs['long_name'] = '1d energy spectrum along y'
    return Ehy


def compute_swt_spectrum(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cH']**2 + uxw['cV']**2 + uxw['cD']**2
    eyw = uyw['cH']**2 + uyw['cV']**2 + uyw['cD']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = '2d energy wavelet scale (2^j) '
    ew.attrs['units'] = 'm^2 s^-2'
    return ew

def compute_swt_horizontal_spectrum(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cH']**2
    eyw = uyw['cH']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = '2d horizontal energy wavelet scale (2^j) '
    ew.attrs['units'] = 'm^2 s^-2'
    return ew

def compute_swt_vertical_spectrum(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cV']**2
    eyw = uyw['cV']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = '2d vertical energy wavelet scale (2^j) '
    ew.attrs['units'] = 'm^2 s^-2'
    return ew

def compute_swt_diagonal_spectrum(ds, dims=['y', 'x'], wtype='haar'):
    uxw = xwl.swt2(ds.ux, levels=ds.attrs['grid.n_lev'], wtype=wtype)
    uyw = xwl.swt2(ds.uy, levels=ds.attrs['grid.n_lev'], wtype=wtype)

    exw = uxw['cD']**2
    eyw = uyw['cD']**2
    ew = exw + eyw
    
    ew.attrs['long_name'] = '2d diagonal energy wavelet scale (2^j) '
    ew.attrs['units'] = 'm^2 s^-2'
    return ew

def compute_1d_swt_spectrum(ds):
    Ew = ds.ew.mean(dim=['y','x'])
    Ew.attrs['long_name'] = '1d swt energy spectrum'
    Ew.attrs['units'] = 'm^2 s^-2'
    return Ew

def compute_1d_swt_spectrum_h(ds):
    Ewh = ds.ewh.mean(dim=['y','x'])
    Ewh.attrs['long_name'] = '1d swt energy spectrum along x'
    Ewh.attrs['units'] = 'm^2 s^-2'
    return Ewh

def compute_1d_swt_spectrum_v(ds):
    Ewv = ds.ewv.mean(dim=['y','x'])
    Ewv.attrs['long_name'] = '1d swt energy spectrum along y'
    Ewv.attrs['units'] = 'm^2 s^-2'
    return Ewv

def compute_jet_coords(ds, dims=['y', 'x'], n_jets=3):
    n_grid = ds.attrs['grid.n_grid']
    n_times = len(ds.time)
    x = ds.x.values
    y = ds.y.values

    y_jet_arr = np.zeros((n_times, n_jets, n_grid), dtype=float)
    y_ajet_arr = np.zeros((n_times, n_jets, n_grid), dtype=float)
    y_njet_arr = np.zeros((n_times, n_jets, n_grid), dtype=float)
    V = np.abs(ds.ux).max().values

    for t in range(n_times):
        for xi in range(n_grid):
            vx = ds.ux.isel(time=t, x=xi).values / V

            vx_periodic = np.concatenate([vx, vx])
            peaks_idx, _ = find_peaks(vx_periodic, distance=n_grid//(1.5*n_jets))
            peaks_idx = peaks_idx[peaks_idx < n_grid]
            top = peaks_idx[np.argsort(vx[peaks_idx])[::-1][:n_jets]]
            if len(top) < n_jets:
                top = np.pad(top, (0, n_jets - len(top)), constant_values=-1)
            jet_idx = np.sort(top)
            y_jet_arr[t, :, xi] = np.where(jet_idx == -1, np.nan, y[np.clip(jet_idx, 0, len(y)-1)])

            avx = -vx
            avx_periodic = np.concatenate([avx, avx])
            peaks_idx, _ = find_peaks(avx_periodic, distance=n_grid//(1.5*n_jets))
            peaks_idx = peaks_idx[peaks_idx < n_grid]
            atop = peaks_idx[np.argsort(avx[peaks_idx])[::-1][:n_jets]]
            if len(atop) < n_jets:
                atop = np.pad(atop, (0, n_jets - len(atop)), constant_values=-1)
            ajet_idx = np.sort(atop)
            y_ajet_arr[t, :, xi] = np.where(ajet_idx == -1, np.nan, y[np.clip(ajet_idx, 0, len(y)-1)])
        
            nvx = 1/np.abs(vx)
            nvx_periodic = np.concatenate([nvx, nvx])
            peaks_idx, _ = find_peaks(nvx_periodic, distance=n_grid//(1.5*n_jets))
            peaks_idx = peaks_idx[peaks_idx < n_grid]
            ntop = peaks_idx[np.argsort(nvx[peaks_idx])[::-1][:n_jets]]
            if len(ntop) < n_jets:
                ntop = np.pad(ntop, (0, n_jets - len(ntop)), constant_values=-1)
            njet_idx = np.sort(ntop)
            y_njet_arr[t, :, xi] = np.where(njet_idx == -1, np.nan, y[np.clip(njet_idx, 0, len(y)-1)])

    y_jet = xr.DataArray(
        y_jet_arr,
        dims=['time', 'k_jet', 'x'],
        coords={'k_jet': np.arange(1, n_jets+1)},
        attrs={'n_jet': n_jets}
    )
    y_ajet = xr.DataArray(
        y_ajet_arr,
        dims=['time', 'k_jet', 'x'],
        coords={'k_jet': np.arange(1, n_jets+1)},
        attrs={'n_jet': n_jets}
    )
    y_njet = xr.DataArray(
        y_njet_arr,
        dims=['time', 'k_jet', 'x'],
        coords={'k_jet': np.arange(1, n_jets+1)},
        attrs={'n_jet': n_jets}
    )
    return y_jet, y_ajet, y_njet

def compute_jet_fspectrum(ds, dims=['y', 'x']):
    y_jet_m = ds.y_jet.mean(dim='x')
    y_ajet_m = ds.y_ajet.mean(dim='x')
    eh_jet = ds.ehx.sel(y=y_jet_m, method='nearest').mean(dim=['k_jet'])
    eh_ajet = ds.ehx.sel(y=y_ajet_m, method='nearest').mean(dim=['k_jet'])
    return eh_jet, eh_ajet


def compute_jet_spectrum(ds, dims=['y', 'x']):
    ew_jet = ds.ew.sel(y=ds.y_jet, method='nearest').mean(dim=['k_jet'])
    ew_ajet = ds.ew.sel(y=ds.y_ajet, method='nearest').mean(dim=['k_jet'])
    ew_njet = ds.ew.sel(y=ds.y_njet, method='nearest').mean(dim=['k_jet'])
    return ew_jet, ew_ajet, ew_njet


def compute_horizontal_jet_spectrum(ds, dims=['y', 'x']):
    ew_jet = ds.ewh.sel(y=ds.y_jet, method='nearest').mean(dim=['k_jet'])
    ew_ajet = ds.ewh.sel(y=ds.y_ajet, method='nearest').mean(dim=['k_jet'])
    ew_njet = ds.ewh.sel(y=ds.y_njet, method='nearest').mean(dim=['k_jet'])
    return ew_jet, ew_ajet, ew_njet

def compute_vertical_jet_spectrum(ds, dims=['y', 'x']):
    ew_jet = ds.ewv.sel(y=ds.y_jet, method='nearest').mean(dim=['k_jet'])
    ew_ajet = ds.ewv.sel(y=ds.y_ajet, method='nearest').mean(dim=['k_jet'])
    ew_njet = ds.ewv.sel(y=ds.y_njet, method='nearest').mean(dim=['k_jet'])
    return ew_jet, ew_ajet, ew_njet

def compute_diagonal_jet_spectrum(ds, dims=['y', 'x']):
    ew_jet = ds.ewd.sel(y=ds.y_jet, method='nearest').mean(dim=['k_jet'])
    ew_ajet = ds.ewd.sel(y=ds.y_ajet, method='nearest').mean(dim=['k_jet'])
    ew_njet = ds.ewd.sel(y=ds.y_njet, method='nearest').mean(dim=['k_jet'])
    return ew_jet, ew_ajet, ew_njet

def calc_diganostic(ds, wtype='haar', n_jets=3):
    n_grid = ds.attrs['grid.n_grid']
    levels = xwl.swt_max_level(n_grid)
    ds.attrs['grid.n_lev'] = levels
    ds['v'] = compute_abs_velocity(ds)
    ds['qx'] = compute_x_vorticity(ds)
    ds['qy'] = compute_y_vorticity(ds)
    ds['vx'] = compute_mean_x_velocity(ds)
    ds['vy'] = compute_mean_y_velocity(ds)
    ds['e_mean'] = compute_mean_energy(ds)
    ds['eh'] = compute_2d_spectrum(ds)
    ds['ehx'] = compute_2d_spectrum_x(ds)
    ds['Eh'] = compute_1d_spectrum(ds)
    ds['Ehx'] = compute_1d_spectrum_x(ds)
    ds['Ehy'] = compute_1d_spectrum_y(ds)
    # ds['ew'] = compute_swt_spectrum(ds, wtype=wtype)
    # ds['ewh'] = compute_swt_horizontal_spectrum(ds, wtype=wtype)
    # ds['ewv'] = compute_swt_vertical_spectrum(ds, wtype=wtype)
    # ds['ewd'] = compute_swt_diagonal_spectrum(ds, wtype=wtype)
    # ds['Ew'] = compute_1d_swt_spectrum(ds)
    # ds['Ewh'] = compute_1d_swt_spectrum_h(ds)
    # ds['Ewv'] = compute_1d_swt_spectrum_v(ds)
    # y_jet, y_ajet, y_njet = compute_jet_coords(ds, n_jets=n_jets)
    # ds['y_jet'] = y_jet
    # ds['y_ajet'] = y_ajet
    # ds['y_njet'] = y_njet
    # eh_jet, eh_ajet = compute_jet_fspectrum(ds)
    # ds['eh_jet'] = eh_jet
    # ds['eh_ajet'] = eh_ajet
    # ew_jet, ew_ajet, ew_njet = compute_jet_spectrum(ds)
    # ds['ew_jet'] = ew_jet
    # ds['ew_ajet'] = ew_ajet
    # ds['ew_njet'] = ew_njet
    # ewh_jet, ewh_ajet, ewh_njet = compute_horizontal_jet_spectrum(ds)
    # ds['ewh_jet'] = ewh_jet
    # ds['ewh_ajet'] = ewh_ajet
    # ds['ewh_njet'] = ewh_njet
    # ewv_jet, ewv_ajet, ewv_njet = compute_vertical_jet_spectrum(ds)
    # ds['ewv_jet'] = ewv_jet
    # ds['ewv_ajet'] = ewv_ajet
    # ds['ewv_njet'] = ewv_njet
    # ewd_jet, ewd_ajet, ewd_njet = compute_diagonal_jet_spectrum(ds)
    # ds['ewd_jet'] = ewd_jet
    # ds['ewd_ajet'] = ewd_ajet
    # ds['ewd_njet'] = ewd_njet

    return ds