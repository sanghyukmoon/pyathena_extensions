from dask.array import fft
import numpy as np
import xarray as xr
from pyathena.util import transform

def fftn(arr):
    assert len(np.unique(arr.chunksizes['x'])) == 1
    assert len(np.unique(arr.chunksizes['y'])) == 1
    assert len(np.unique(arr.chunksizes['z'])) == 1
    chunksizes = np.array((arr.chunksizes['z'][0], arr.chunksizes['y'][0], arr.chunksizes['x'][0]))
    num_chunks = arr.shape // chunksizes
    nc1, nc2 = closest_factors(num_chunks.prod())
    arr = arr.transpose('z', 'y', 'x').data
    arr = arr.rechunk((arr.shape[0]//nc1, arr.shape[1]//nc2, arr.shape[2]))
    arr = fft.fft(arr, axis=2)
    arr = arr.rechunk((arr.shape[0]//nc1, arr.shape[1], arr.shape[2]//nc2))
    arr = fft.fft(arr, axis=1)
    arr = arr.rechunk((arr.shape[0], arr.shape[1]//nc1, arr.shape[2]//nc2))
    arr = fft.fft(arr, axis=0)
    arr = arr.rechunk(chunksizes)
    return arr

def power_spectrum(arr, nx, lbox, nbin=50):
    # Perform FFT and calculate power spectrum
    arr_k = (lbox/nx)**3*fftn(arr)
    pspec = np.abs(arr_k)**2 / lbox**3
    kx = 2*np.pi*fft.fftfreq(nx, d=lbox/nx)
    pspec = xr.DataArray(pspec, coords=dict(kz=kx, ky=kx, kx=kx))
    kx, ky, kz = transform._chunk_like(pspec.kx, pspec.ky, pspec.kz, chunks=pspec.chunksizes)
    pspec.coords['kmag'] = np.sqrt(kz**2 + ky**2 + kx**2)
    kmin = 2*np.pi/lbox
    kmax = np.pi/(lbox/nx)
    pspec_avg = transform.groupby_bins(pspec, 'kmag', nbin, (kmin, kmax))
    return pspec_avg

def generate_grf(nx, lbox, mean, varience, power_index):
    from scipy.stats import gumbel_r
    from scipy import fft as spfft
    # Create wavenumber grid
    kx = 2*np.pi*spfft.fftfreq(nx, d=lbox/nx)
    kmag = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kx[None, None, :]**2)

    # Set up power spectrum from the input varience and power_index
    nonzero_mask = (kmag != 0)
    power_spectrum = np.zeros_like(kmag)
    power_spectrum[nonzero_mask] = kmag[nonzero_mask]**power_index

    white_noise = np.random.normal(size=(nx, nx, nx))
#    y = gumbel_r()
#    white_noise = -y.rvs((nx,nx,nx))
    white_noise_k = spfft.fftn(white_noise)
    fourier_coeff = white_noise_k * np.sqrt(power_spectrum)
    grf = spfft.ifftn(fourier_coeff).real
    grf *= np.sqrt((varience / grf.var()))
    grf += mean
    return grf

def closest_factors(n: int) -> tuple[int, int]:
    """
    Factors a given positive integer into its two closest integer factors.

    This function finds two integers, a and b, such that a * b = n
    and the absolute difference |a - b| is minimized.

    Args:
        n: A positive integer to be factored.

    Returns:
        A tuple containing the two closest factors (a, b) sorted such that a <= b.

    Raises:
        ValueError: If the input number is not a positive integer.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    n = int(n)

    # Start searching from the integer part of the square root of n
    start_point = int(np.sqrt(n))

    # Iterate downwards from the starting point to 1
    for i in range(start_point, 0, -1):
        # Check if i is a factor of n
        if n % i == 0:
            factor1 = i
            factor2 = n // i
            # The first factor found gives the pair with the smallest difference
            return (factor1, factor2)

    # This part of the code is technically unreachable for n > 0,
    # as 1 is always a factor. It's included for logical completeness.
    # It would be reached only if the loop failed, which it won't.
    return (1, n)
