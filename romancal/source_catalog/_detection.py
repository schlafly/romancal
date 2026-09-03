"""
Module to detect sources using image segmentation.
"""

import logging
import math
import warnings


import numpy as np
import scipy.signal
from astropy.convolution import convolve
from astropy.utils.exceptions import AstropyUserWarning
from photutils.segmentation import SourceFinder, make_2dgaussian_kernel
from photutils.utils.exceptions import NoDetectionsWarning

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def convolve_data(data, kernel_fwhm, size=None, mask=None):
    """
    Convolve the background-subtracted model image with a Gaussian
    kernel.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        The input 2D array. The data is assumed to be
        background subtracted.
    kernel_fwhm : float
        The FWHM of the Gaussian kernel.
    size : int, optional
        The size of the kernel array. Default is 3 times the kernel
        FWHM.
    mask : 2D `numpy.ndarray`, optional
        A boolean mask with the same shape as the input data. True
        values indicate masked pixels.

    Returns
    -------
    convolved_data : `numpy.ndarray`
        The convolved data array.
    """
    size = math.ceil(kernel_fwhm * 3)
    size = size + 1 if size % 2 == 0 else size  # make size be odd
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=size)  # normalized to 1

    with warnings.catch_warnings():
        # suppress warnings caused by large masked areas
        warnings.simplefilter("ignore", AstropyUserWarning)
        return convolve(data, kernel, mask=mask)


def make_segmentation_image(
    convolved_data, snr_threshold, n_pixels, bkg_rms, deblend=False, mask=None,
    deblend_contrast=1e-4,
):
    """
    Make a segmentation image from a model image.

    Parameters
    ----------
    convolved_data : 2D `numpy.ndarray`
        The background-subtracted convolved data array.
    snr_threshold : float
        The per-pixel signal-to-noise ratio threshold for detection.
    n_pixels : int
        The number of connected pixels required to define a source.
    bkg_rms : 2D `numpy.ndarray`
        The background RMS array.
    deblend : bool, optional
        Whether to deblend overlapping sources. Default is False.
    mask : 2D `numpy.ndarray`, optional
        A boolean mask with the same shape as the input data. True
        values indicate masked pixels.
    deblend_contrast : float, optional
        The fraction of the total source flux that a local peak must have to
        be deblended as a separate source.

    Returns
    -------
    segment_img : `SegmentationImage`
        The segmentation image.
    """
    with warnings.catch_warnings():
        # suppress NoDetectionsWarning from photutils
        warnings.filterwarnings("ignore", category=NoDetectionsWarning)

        finder = SourceFinder(
            n_pixels, deblend=deblend, contrast=deblend_contrast
        )
        threshold = snr_threshold * bkg_rms
        segment_img = finder(convolved_data, threshold, mask=mask)

        if segment_img is None:
            n_sources = 0
        else:
            n_sources = segment_img.n_labels
        log.info(f"Detected {n_sources} sources")

    return segment_img


# Photutils discretizes the model on a grid oversampled 10x in each axis.  For
# a 601 px kernel that is a 6010^2 grid and two gigabytes -- the single largest
# allocation in the step, larger than any of the convolutions it feeds.  The
# refinement only earns its keep where a pixel spans real curvature, so bound
# the oversampled grid instead: the small kernels keep the full factor of ten
# and come out bit-identical, and the large ones shift by a few parts in 10^6.
_MAX_OVERSAMPLED_GRID = 2000


def _oversampling(size):
    """Oversampling factor for a kernel of ``size`` pixels across."""
    return int(np.clip(_MAX_OVERSAMPLED_GRID // size, 1, 10))


def make_gaussian_kernel(fwhm, lower=False, size_factor=3, max_size=601):
    """
    Make a normalized 2D circular Gaussian kernel.

    Parameters
    ----------
    fwhm : float
        Full-width at half-maximum of the Gaussian, in pixels.
    size_factor : int, optional
        Kernel box size as a multiple of ``fwhm``.  It does double duty: it
        sets how far into the Gaussian's wings the template reaches, and,
        when ``lower`` is set, it is also the scale of the background being
        removed, since the mean is taken over this same box.  Reaching the
        wings needs only a few FWHM; removing a background needs a box wide
        enough to hold the structure being removed, which is much larger, so
        a lowered kernel wants the larger value.  The default of 3 matches
        the convention used elsewhere in the pipeline.
    lower : bool, optional
        If `True`, return a "lowered" (zero-sum) kernel

            k' = k - mean(k)

        i.e. the matched filter for a source plus an unknown background
        constant over the kernel box.  A zero-sum kernel gives no response
        to a flat background, so a compact source riding on a larger
        object's light keeps only the smaller source's addition over the
        background.
    max_size : int, optional
        Largest kernel size, in pixels.

    Returns
    -------
    kernel : 2D `numpy.ndarray`
        The kernel array.  Sums to 1, or to 0 when ``lower`` is set.
    """
    size = min(math.ceil(size_factor * fwhm), max_size)
    size = size + 1 if size % 2 == 0 else size  # make size be odd
    kernel = np.asarray(  # sums to 1
        make_2dgaussian_kernel(fwhm, size=size,
                               oversampling=_oversampling(size))
    )

    if lower:
        kernel = kernel - kernel.mean()

    return kernel


def fft_convolve(data, kernel):
    """
    Convolve ``data`` with ``kernel``, zero-padded at the boundary.

    Single precision throughout: the transforms are the bulk of this step's
    memory, and against a double-precision transform the results agree to
    seven digits, a millionth of the noise.  Non-finite values are zeroed
    first, since an FFT would otherwise spread them over the whole frame.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        The array to convolve.
    kernel : 2D `numpy.ndarray`
        The convolution kernel, with odd sides so that "same" is centered.

    Returns
    -------
    convolved : 2D `numpy.ndarray`
        The convolved array, of the same shape as ``data``.
    """
    data = np.nan_to_num(np.asarray(data, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.asarray(kernel, dtype=np.float32)
    return scipy.signal.fftconvolve(data, kernel, mode="same")


def flux_convolve(data, kernel, mask=None):
    """
    Unweighted convolution of the data with a template.

    This is the image centroids and moments are measured on.  Kept separate
    from `ivw_convolve` because a caller wants one or the other, never both,
    and each convolution is expensive.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        Background-subtracted data.
    kernel : 2D `numpy.ndarray`
        The convolution kernel.
    mask : 2D `numpy.ndarray`, optional
        Boolean mask; True values force the data to zero.

    Returns
    -------
    conv_flux : 2D `numpy.ndarray`
        ``conv(data, kernel)``.
    """
    if mask is not None:
        data = np.where(mask, 0.0, data)
    return fft_convolve(data, kernel)


def ivw_convolve(data, wht, kernel, mask=None):
    """
    Inverse-variance-weighted convolution of data with uncertainties given a template.

    Construction of a significance image for a matched filter requires
    a signal and uncertainty image for each template.  This function computes
    both.  The ratio ``num / sqrt(denom2)`` returned by this function is the
    maximum-likelihood amplitude of a template kernel divided by its uncertainty,
    given some Gaussian noise.

    Masked pixels enter with zero weight and are not excluded,
    so the result is defined on masked pixels.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        Background-subtracted data.
    wht : 2D `numpy.ndarray`
        Inverse-variance weights; zero where masked.
    kernel : 2D `numpy.ndarray`
        The convolution kernel.
    mask : 2D `numpy.ndarray`, optional
        Boolean mask; True values force the data and weights to zero.

    Returns
    -------
    num : 2D `numpy.ndarray`
        ``conv(data * wht, kernel)``.
    denom2 : 2D `numpy.ndarray`
        ``conv(wht, kernel**2)``.
    """
    if mask is not None:
        data = np.where(mask, 0.0, data)
        wht = np.where(mask, 0.0, wht)

    return fft_convolve(data * wht, kernel), fft_convolve(wht, kernel**2)


def snr_from_ivw(num, denom2):
    """
    Form a signal-to-noise ratio image from accumulated `ivw_convolve` arrays.

    Kept separate from `ivw_convolve` because the arrays must be summed over
    bands before the ratio is taken.  ``denom2`` is mathematically
    non-negative, but the FFT returns small negative values where it should
    return zero, so it is clamped before the square root.

    Parameters
    ----------
    num, denom2 : 2D `numpy.ndarray`
        The numerator and squared-denominator arrays, summed over bands.

    Returns
    -------
    snr : 2D `numpy.ndarray`
        ``num / sqrt(denom2)``, zero where there is no weight.
    """
    denom = np.sqrt(np.maximum(denom2, 0.0))
    return np.where(denom > 0, num / denom, 0.0)
