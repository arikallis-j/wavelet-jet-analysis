import numpy as np
from pyqg import QParameterization

class RingForcing(QParameterization):
    r"""Stochastically force PV in spectral space on a ring
    associated with a band of given wavenumbers from 
    `Uchida et al. (2023)`_.

    This parametrization introduces a vertically uniform stochastic
    forcing decorrelated in time by inverse Fourier transforming
    white noise in the wavenumber domain

    .. math:: \hat{w}(t,k^y,k^x) = a(t,k^y,k^x) + ib(t,k^y,k^x)

    in the wavenumber band between 
    :math:`k_{in}<\sqrt{{k^x}^2 + {k^y}^2}<k_{out}`
    and is zero outside of this band.
    :math:`k^x, k^y` are the zonal and meridional 
    wavenumbers and :math:`a, b` are Gaussian random variables.
    
    After taking the inverse Fourier transform, the horizontal 
    mean is removed and then divided by the horizontal mean 
    of the absolute values to have the amplitudes on the 
    order of unity.

    The magnitude of the noise can be adjusted by the input variable
    `mag_noise_forc`.

    .. _Uchida et al. (2023): https://doi.org/10.31223/X5C063
    """

    def __init__(self, k_in_forc=0, k_out_forc=0, mag_noise_forc=0, layers='all'):
        r"""
        Parameters
        ----------
        k_in_forc : number
            Inner wave number of the ring
            Defaults to 0.0.
        k_out_forc : number
            Outer wave number of the ring
            Defaults to 0.0.
        mag_noise_forc : number
            Amplitude of the forcing 
        """

        self.k_in_forc = k_in_forc
        self.k_out_forc = k_out_forc
        self.mag_noise_forc = mag_noise_forc
        self.layers = layers

    def __call__(self, m):
        nhx,nhy = m.wv.shape
        wvx = np.sqrt((m.k)**2.+(m.l)**2.)
        
        mask = np.ones_like(wvx)
        mask[wvx<=self.k_in_forc] = 0.
        mask[wvx>self.k_out_forc] = 0.

        Ring_hat = mask*(np.random.randn(nhx,nhy) +1j*np.random.randn(nhx,nhy))
        
        if self.layers == 'all':
            Ring = m.ifft( Ring_hat[np.newaxis,:,:] )
        else:
            Ring = np.zeros_like(m.u)
            if self.layers == 'surf':
                Ring[0] = m.ifft( Ring_hat[np.newaxis,:,:] )[0]
            elif self.layers == 'bottom':
                Ring[-1] = m.ifft( Ring_hat[np.newaxis,:,:] )[-1]
        
        Ring = Ring - Ring.mean(axis=(-1,-2))[:,np.newaxis,np.newaxis]
        Ring = Ring / np.abs(Ring).mean(axis=(-1,-2))[:,np.newaxis,np.newaxis] 

        dq = self.mag_noise_forc*Ring

        return dq

    def __repr__(self):
        return f"RingForcing(k_in_forc={self.k_in_forc}, "\
                           f"k_out_forc={self.k_out_forc})"