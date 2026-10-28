import numpy as np
import xarray as xr
import xrft

#############
# Mathematics
#############

def gradient(v, dim=['y', 'x']):
    vh = xrft.fft(v, dim=dim)
    lag_y, lag_x = vh.freq_y.attrs.get('direct_lag', 0.0),  vh.freq_x.attrs.get('direct_lag', 0.0)
    kx, ky = 2 * np.pi * vh.freq_x, 2 * np.pi * vh.freq_y
    gradx_vh = 1j * vh * kx
    grady_vh = 1j * vh * ky 
    gradx_v = xrft.ifft(gradx_vh, dim=['freq_y', 'freq_x'], lag=[lag_y, lag_x]).real
    grady_v = xrft.ifft(grady_vh, dim=['freq_y', 'freq_x'], lag=[lag_y, lag_x]).real
    return gradx_v, grady_v

def spectral_density(v, dim=['y', 'x']):
    vh = xrft.fft(v, dim=dim)
    rh = 0.5 * (np.abs(vh)**2)
    rh = rh.assign_coords({
        'freq_x': 2 * np.pi * rh.freq_x,
        'freq_y': 2 * np.pi * rh.freq_y,
    }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
    return rh

def spectral_transfer(v, ux, uy, dim=['y', 'x']):
    gradx_v, grady_v = gradient(v, dim=dim)
    adv_v = ux * gradx_v.values +  uy * grady_v.values
    vh = xrft.fft(v, dim=dim)
    adv_vh = xrft.fft(adv_v, dim=dim)
    tauh = - np.real(np.conj(vh) * adv_vh)
    tauh = tauh.assign_coords({
        'freq_x': 2 * np.pi * tauh.freq_x,
        'freq_y': 2 * np.pi * tauh.freq_y,
    }).rename({'freq_x': 'kx', 'freq_y': 'ky'})
    return tauh

def radial_component(vh, fftdim=['ky', 'kx'], nfactor=4):
    Vhr = xrft.isotropize(vh, fftdim=fftdim, nfactor=nfactor)
    Vhr = Vhr.rename({'freq_r': 'k_r'})
    return Vhr

def zonal_component(vh, fftdim=['ky', 'kx']):
    Vhz = vh.sel(ky=0, method='nearest')
    return Vhz

def meridional_component(vh, fftdim=['ky', 'kx']):
    Vhm = vh.sel(kx=0, method='nearest')
    return Vhm


################
# Energy Spectra
################

def compute_energy_spectral_density(ds, dim=['y', 'x']):
    ehx = spectral_density(ds.ux, dim=dim)
    ehy = spectral_density(ds.uy, dim=dim)
    eh = ehx + ehy
    eh.attrs['long_name'] = 'energy spectral density'
    eh.attrs['units'] = 'm^2 s^-2 / m^-2'
    return eh

def compute_radial_energy_spectral_density(ds, fftdim=['ky', 'kx'], nfactor=4):
    Ehr = radial_component(ds.eh, fftdim=fftdim, nfactor=nfactor)
    Ehr.attrs['long_name'] = 'radial energy spectral density'
    Ehr.attrs['units'] = 'm^2 s^-2 / m^-2'
    return Ehr

def compute_zonal_energy_spectral_density(ds, fftdim=['ky', 'kx']):
    Ehz = zonal_component(ds.eh, fftdim=fftdim)
    Ehz.attrs['long_name'] = 'zonal energy spectral density'
    Ehz.attrs['units'] = 'm^2 s^-2 / m^-2'
    return Ehz

def compute_meridional_energy_spectral_density(ds, fftdim=['ky', 'kx']):
    Ehm = meridional_component(ds.eh, fftdim=fftdim)
    Ehm.attrs['long_name'] = 'meridional energy spectral density'
    Ehm.attrs['units'] = 'm^2 s^-2 / m^-2'
    return Ehm

def calc_energy_spectra(ds):
    ds['eh'] = compute_energy_spectral_density(ds)
    ds['Ehr'] = compute_radial_energy_spectral_density(ds)
    ds['Ehz'] = compute_zonal_energy_spectral_density(ds)
    ds['Ehm'] = compute_meridional_energy_spectral_density(ds)
    return ds

###################
# Enstrophy Spectra
###################

def compute_enstrophy_spectral_density(ds, dim=['y', 'x']):
    zh = spectral_density(ds.q, dim=dim)
    zh.attrs['long_name'] = 'enstrophy spectral density'
    zh.attrs['units'] = 's^-2 / m^-2'
    return zh

def compute_radial_enstrophy_spectral_density(ds, fftdim=['ky', 'kx'], nfactor=4):
    Zhr = radial_component(ds.zh, fftdim=fftdim, nfactor=nfactor)
    Zhr.attrs['long_name'] = 'radial enstrophy spectral density'
    Zhr.attrs['units'] = 's^-2 / m^-2'
    return Zhr

def compute_zonal_enstrophy_spectral_density(ds, fftdim=['ky', 'kx']):
    Zhz = zonal_component(ds.zh, fftdim=fftdim)
    Zhz.attrs['long_name'] = 'zonal enstrophy spectral density'
    Zhz.attrs['units'] = 's^-2 / m^-2'
    return Zhz

def compute_meridional_enstrophy_spectral_density(ds, fftdim=['ky', 'kx']):
    Zhm = meridional_component(ds.zh, fftdim=fftdim)
    Zhm.attrs['long_name'] = 'meridional enstrophy spectral density'
    Zhm.attrs['units'] = 's^-2 / m^-2'
    return Zhm

def calc_enstrophy_spectra(ds):
    ds['zh'] = compute_enstrophy_spectral_density(ds)
    ds['Zhr'] = compute_radial_enstrophy_spectral_density(ds)
    ds['Zhz'] = compute_zonal_enstrophy_spectral_density(ds)
    ds['Zhm'] = compute_meridional_enstrophy_spectral_density(ds)
    return ds

#################
# Energy Transfer
#################

def compute_energy_spectral_transfer(ds, dim=['y', 'x']):
    epshx = spectral_transfer(ds.ux, ds.ux, ds.uy, dim=['y', 'x'])
    epshy = spectral_transfer(ds.uy, ds.ux, ds.uy, dim=['y', 'x'])
    epsh = epshx + epshy
    epsh.attrs['long_name'] = 'energy spectral transfer'
    epsh.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return epsh

def compute_radial_energy_spectral_transfer(ds, fftdim=['ky', 'kx'], nfactor=4):
    Epshr = radial_component(ds.epsh, fftdim=fftdim, nfactor=nfactor)
    Epshr.attrs['long_name'] = 'radial energy spectral transfer'
    Epshr.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return Epshr

def compute_zonal_energy_spectral_transfer(ds, fftdim=['ky', 'kx']):
    Epshz = zonal_component(ds.epsh, fftdim=fftdim)
    Epshz.attrs['long_name'] = 'zonal energy spectral transfer'
    Epshz.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return Epshz

def compute_meridional_energy_spectral_transfer(ds, fftdim=['ky', 'kx']):
    Epshm = meridional_component(ds.epsh, fftdim=fftdim)
    Epshm.attrs['long_name'] = 'meridional energy spectral transfer'
    Epshm.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return Epshm

def calc_energy_transfer(ds):
    ds['epsh'] = compute_energy_spectral_transfer(ds)
    ds['Epshr'] = compute_radial_energy_spectral_transfer(ds)
    ds['Epshz'] = compute_zonal_energy_spectral_transfer(ds)
    ds['Epshm'] = compute_meridional_energy_spectral_transfer(ds)
    return ds

####################
# Enstrophy Transfer
####################

def compute_enstrophy_spectral_transfer(ds, dim=['y', 'x']):
    etah = spectral_transfer(ds.q, ds.ux, ds.uy, dim=['y', 'x'])
    etah.attrs['long_name'] = 'enstrophy spectral transfer'
    etah.attrs['units'] = 's^-2 / (s m^-2)'
    return etah

def compute_radial_enstrophy_spectral_transfer(ds, fftdim=['ky', 'kx'], nfactor=4):
    Etahr = radial_component(ds.epsh, fftdim=fftdim, nfactor=nfactor)
    Etahr.attrs['long_name'] = 'radial enstrophy spectral transfer'
    Etahr.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return Etahr

def compute_zonal_enstrophy_spectral_transfer(ds, fftdim=['ky', 'kx']):
    Etahz = zonal_component(ds.epsh, fftdim=fftdim)
    Etahz.attrs['long_name'] = 'zonal enstrophy spectral transfer'
    Etahz.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return Etahz

def compute_meridional_enstrophy_spectral_transfer(ds, fftdim=['ky', 'kx']):
    Etahm = meridional_component(ds.epsh, fftdim=fftdim)
    Etahm.attrs['long_name'] = 'meridional enstrophy spectral transfer'
    Etahm.attrs['units'] = 'm^2 s^-2 / (s m^-2)'
    return Etahm

def calc_enstrophy_transfer(ds):
    ds['etah'] = compute_enstrophy_spectral_transfer(ds)
    ds['Etahr'] = compute_radial_enstrophy_spectral_transfer(ds)
    ds['Etahz'] = compute_zonal_enstrophy_spectral_transfer(ds)
    ds['Etahm'] = compute_meridional_enstrophy_spectral_transfer(ds)
    return ds

def calc_fourier(ds):
    ds = calc_energy_spectra(ds)
    ds = calc_enstrophy_spectra(ds)
    ds = calc_energy_transfer(ds)
    ds = calc_enstrophy_transfer(ds)
    return ds