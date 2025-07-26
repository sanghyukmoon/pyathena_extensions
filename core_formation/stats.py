from scipy import fft
import numpy as np
import xarray as xr
from pyathena.util import transform

def power_spectrum(arr, nx, lbox, nbin=50):
    # Perform FFT and calculate power spectrum
    arr_k = (lbox/nx)**3*fft.fftn(arr)
    pspec = np.abs(arr_k)**2 / lbox**3
    kx = 2*np.pi*fft.fftfreq(nx, d=lbox/nx)
    pspec = xr.DataArray(pspec, coords=dict(kz=kx, ky=kx, kx=kx))
    pspec.coords['kmag'] = np.sqrt(pspec.kz**2 + pspec.ky**2 + pspec.kx**2)
    pspec_avg = transform.groupby_bins(pspec, 'kmag', nbin, (np.abs(kx).min(), np.abs(kx).max()))
    return pspec_avg

def generate_grf(nx, lbox, mean, varience, power_index):
    # Create wavenumber grid
    kx = 2*np.pi*fft.fftfreq(nx, d=lbox/nx)
    kmag = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kx[None, None, :]**2)

    # Set up power spectrum from the input varience and power_index
    nonzero_mask = (kmag != 0)
    power_spectrum = np.zeros_like(kmag)
    power_spectrum[nonzero_mask] = kmag[nonzero_mask]**power_index

    white_noise = np.random.normal(size=(nx, nx, nx))
    white_noise_k = fft.fftn(white_noise)
    fourier_coeff = white_noise_k * np.sqrt(power_spectrum)
    grf = fft.ifftn(fourier_coeff).real
    grf *= np.sqrt((varience / grf.var()))
    grf += mean
    return grf
