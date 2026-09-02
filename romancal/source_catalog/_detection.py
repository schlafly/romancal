"""
Module to detect sources using image segmentation.
"""

import logging
import math
import warnings

import scipy.signal

import numpy as np
from astropy.convolution import convolve, convolve_fft
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
    convolved_data, snr_threshold, n_pixels, bkg_rms, deblend=False, mask=None
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

    Returns
    -------
    segment_img : `SegmentationImage`
        The segmentation image.
    """
    with warnings.catch_warnings():
        # suppress NoDetectionsWarning from photutils
        warnings.filterwarnings("ignore", category=NoDetectionsWarning)

        finder = SourceFinder(n_pixels, deblend=deblend, contrast=1e-4)
        threshold = snr_threshold * bkg_rms
        segment_img = finder(convolved_data, threshold, mask=mask)

        if segment_img is None:
            n_sources = 0
        else:
            n_sources = segment_img.n_labels
        log.info(f"Detected {n_sources} sources")

    return segment_img


def make_gaussian_kernel(fwhm, lower_scale=0.0, size_factor=3,
                         max_size=601):
    """
    Make a normalized 2D circular Gaussian kernel.

    Parameters
    ----------
    fwhm : float
        Full-width at half-maximum of the Gaussian, in pixels.
    size_factor : int, optional
        Kernel box size as a multiple of ``fwhm`` (before any enlargement
        for ``lower_scale``).  The default of 3 matches the convention used
        elsewhere in the pipeline; the template detection uses 4 so that a
        lowered kernel's negative surround is not truncated.
    lower_scale : float, optional
        If greater than zero, return a "lowered" (zero-sum) kernel

            k' = k - k * g,   FWHM(g) = lower_scale * fwhm

        i.e. the matched filter for a source plus an unknown background
        varying on ``lower_scale`` times the template's own scale.  A
        zero-sum kernel gives no response to a flat background, so a
        compact source riding on a larger object's light keeps only its
        prominence above that light.  The box is enlarged to hold the
        resulting negative surround, capped at ``max_size``.
    max_size : int, optional
        Largest kernel size, in pixels.

    Returns
    -------
    kernel : 2D `numpy.ndarray`
        The kernel array.  Sums to 1, or to 0 when ``lower_scale > 0``.
    """
    size = math.ceil(size_factor * fwhm)
    if lower_scale > 0:
        size = min(math.ceil(size_factor * fwhm * (1.0 + lower_scale)),
                   max_size)
    size = size + 1 if size % 2 == 0 else size  # make size be odd
    kernel = np.asarray(make_2dgaussian_kernel(fwhm, size=size))  # sums to 1

    if lower_scale > 0:
        gsize = min(math.ceil(size_factor * lower_scale * fwhm) | 1, size)
        smooth = np.asarray(make_2dgaussian_kernel(lower_scale * fwhm,
                                                   size=gsize))
        smooth = smooth / smooth.sum()
        kernel = kernel - scipy.signal.fftconvolve(kernel, smooth, mode="same")

    return kernel


def ivw_convolve(data, wht, kernel, mask=None):
    """
    Inverse-variance-weighted convolution: the matched-filter building block.

    Returns the arrays that must be accumulated (across bands, if there is
    more than one) before the signal-to-noise ratio can be formed.  The
    weighted form ``conv(data * wht, k) / sqrt(conv(wht, k^2))`` is the
    maximum-likelihood amplitude of ``k`` divided by its uncertainty, for
    spatially varying Gaussian noise.  Unlike
    ``conv(data, k) / sqrt(conv(var, k^2))`` it down-weights noisy pixels
    independently in the numerator and the denominator, so one high-variance
    pixel cannot inflate the denominator across a whole footprint.

    Because masked pixels enter with zero weight rather than being excluded,
    the result is defined wherever *any* weight falls under the kernel --
    including at masked pixels themselves.

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
    conv_flux : 2D `numpy.ndarray`
        ``conv(data, kernel)``, unweighted; for centroids and moments.
    """
    kwargs = {
        "normalize_kernel": False,
        "allow_huge": True,
        "boundary": "fill",
        "fill_value": 0.0,
    }
    if abs(float(kernel.sum())) < 1e-8:
        # a zero-sum kernel cannot be normalized, so astropy's default
        # nan_treatment="interpolate" is undefined for it; NaNs are already
        # zeroed through ``mask``, so filling them is equivalent
        kwargs["nan_treatment"] = "fill"
    if mask is not None:
        data = np.where(mask, 0.0, data)
        wht = np.where(mask, 0.0, wht)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        num = convolve_fft(data * wht, kernel, **kwargs)
        denom2 = convolve_fft(wht, kernel**2, **kwargs)
        conv_flux = convolve_fft(data, kernel, **kwargs)

    return num, denom2, conv_flux


def snr_from_ivw(num, denom2):
    """
    Form a signal-to-noise ratio image from accumulated `ivw_convolve` arrays.

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
