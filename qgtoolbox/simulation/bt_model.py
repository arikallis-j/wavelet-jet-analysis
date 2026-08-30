import pyqg
import numpy as np
from pyqg import Smagorinsky
from .ring_forcing import RingForcing

def make_bt_model(config, log_level=3):
    # architechure parameters
    n_threads = config['architecture']['n_threads']

    # grid parameters
    n_grid =  config['grid']['n_grid']

    # drag model
    r_ek = config['model']['rek']

    # viscocity model
    C_nu = config['model']['cnu']
    viscocity = Smagorinsky(constant=C_nu)

    # forcing model
    Af, kf, dkf = config['model']['af'], config['model']['kf'], config['model']['dkf']
    k_in, k_out = kf - dkf, kf + dkf
    forcing = RingForcing(k_in_forc=k_in, k_out_forc=k_out, mag_noise_forc=Af)

    # beta model
    beta = config['model']['beta']

    # simulation parameters
    dt, t_write, t_epoch = config['simulation']['dt'], config['simulation']['t_write'], config['simulation']['t_epoch']

    model = pyqg.BTModel(
        L = 2*np.pi, nx=n_grid,
        rek = r_ek, beta = beta,
        uv_parameterization = viscocity,
        q_parameterization = forcing,
        dt = dt, tmax = t_epoch,
        ntd = n_threads,
        log_level = log_level,
        twrite = t_write,
    )

    model.set_q(np.zeros((1, model.ny,model.nx)))

    return model