"""
Module to detect sources with a bank of matched filters.

The image is convolved with several templates spanning a range of source
sizes.  Each convolution is turned into a significance image -- the
maximum-likelihood amplitude of that template divided by its uncertainty --
and the per-pixel maximum over templates becomes a single detection image on
which sources are found and deblended exactly once.

Two properties make this work.  Every local maximum of the maximum image is a
local maximum of whichever template attains it there (if template ``k`` wins
at ``x*`` then for nearby ``x``, ``D_k(x) <= M(x) <= M(x*) = D_k(x*)``), so
taking the maximum invents no peaks.  And because the templates are
"lowered" -- built to sum to zero -- none of them responds to a smooth
background, so a compact source sitting on a large galaxy's light is measured
by its prominence above that light rather than by the light itself.
"""

import logging

import numpy as np
import scipy.ndimage
from photutils.segmentation import SegmentationImage, deblend_sources

from romancal.source_catalog._background import RomanBackground
from romancal.source_catalog._detection import (
    ivw_convolve,
    make_gaussian_kernel,
    make_segmentation_image,
    snr_from_ivw,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Template bank.  The first entry is the PSF and is sized by the step's
# ``kernel_fwhm``; the rest are exponential discs of half-light radius 4, 16
# and 64 pixels, converted to the equivalent Gaussian FWHM (FWHM ~ 1.68 r_e).
# The spacing is logarithmic so that a source of any size is within a factor
# of two of some template.
_TEMPLATE_RE = (4.0, 16.0, 64.0)  # exponential half-light radii, pixels
_RE_TO_FWHM = 1.678

# Kernel box size, in units of the template FWHM.
_SIZE_FACTOR = 4

# Background scale for the lowered templates, in units of the template FWHM.
# The background a template must ignore is set by the *contaminant* -- a
# galaxy's extended light -- not by the template, so this is deliberately
# large: subtracting on the template's own scale makes a marginally resolved
# source its own background and destroys it (the PSF template keeps 94% of a
# point source but only 55% of a source of FWHM 6).  At 10 the whole bank
# loses about 1.6% of its norm.
_LOWER_SCALE = 10.0

# Deblending contrast for the maximum image.
_DEBLEND_CONTRAST = 0.001

# Grow each final segment by this many pixels.  At a threshold of 5 sigma a
# source near the limit covers a single pixel, and moments need its
# neighbours; one dilation turns that into a 3x3.
_SEGMENT_DILATE = 1

# Smallest final segment to keep, counted in pixels the photometry can
# actually use.  Dilation grows a lone pixel to a full 3x3, so 9 means "the
# source must retain a complete, unmasked, unclaimed neighbourhood".  Fewer
# pixels give degenerate second moments and a non-finite centroid.
_MIN_SEGMENT_PIXELS = 9


def _template_fwhms(kernel_fwhm):
    """Template FWHMs in pixels, PSF first."""
    return (float(kernel_fwhm),) + tuple(_RE_TO_FWHM * re for re in _TEMPLATE_RE)


def make_template_snr_images(data, err, kernel_fwhm, mask=None):
    """
    Compute a matched-filter significance image for each template.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        Background-subtracted data.
    err : 2D `numpy.ndarray`
        Per-pixel uncertainty.
    kernel_fwhm : float
        FWHM of the PSF template, in pixels.
    mask : 2D `numpy.ndarray`, optional
        Boolean mask; True values are given zero weight.

    Returns
    -------
    snr_images : list of 2D `numpy.ndarray`
        One matched-filter SNR image per template.
    conv_psf : 2D `numpy.ndarray`
        The PSF-convolved flux image, for centroids and moments.  This uses
        the *unlowered* PSF kernel: a zero-sum kernel produces a negative
        surround that makes moments meaningless.
    """
    good = np.ones(data.shape, dtype=bool) if mask is None else ~mask
    wht = np.where(good, 1.0 / np.where(good, err, 1.0) ** 2, 0.0)

    snr_images = []
    for fwhm in _template_fwhms(kernel_fwhm):
        kernel = make_gaussian_kernel(
            fwhm, lower_scale=_LOWER_SCALE, size_factor=_SIZE_FACTOR
        )
        if kernel.shape[0] > min(data.shape):
            # a kernel wider than the image is all boundary: the lowered
            # kernel's negative surround falls off the edge, so it no longer
            # sums to zero over the data and responds to the background.
            # Real images are far larger than the bank, but cutouts and test
            # frames are not.
            log.info(
                f"Skipping template FWHM={fwhm:.1f} px: kernel "
                f"{kernel.shape[0]} px exceeds the image"
            )
            continue
        num, denom2, _ = ivw_convolve(data, wht, kernel, mask=mask)
        snr_images.append(snr_from_ivw(num, denom2))
        log.info(f"Template FWHM={fwhm:.1f} px: kernel {kernel.shape[0]} px")

    if not snr_images:
        msg = (
            f"No template fits within an image of shape {data.shape}; the "
            f"smallest kernel is "
            f"{make_gaussian_kernel(kernel_fwhm, lower_scale=_LOWER_SCALE, size_factor=_SIZE_FACTOR).shape[0]} px"
        )
        raise ValueError(msg)

    psf_kernel = make_gaussian_kernel(kernel_fwhm, size_factor=_SIZE_FACTOR)
    _, _, conv_psf = ivw_convolve(data, wht, psf_kernel, mask=mask)

    return snr_images, conv_psf


def _max_detection_image(snr_images, thresholds):
    """
    Per-pixel maximum over templates, each in units of its own threshold.

    Dividing by the threshold makes the templates comparable, so ``M > 1``
    means "some template detects this pixel" and the winning template is the
    one with the highest significance.

    Returns
    -------
    max_image : 2D `numpy.ndarray`
        The maximum image.
    winner : 2D `numpy.ndarray`
        Index of the template attaining the maximum at each pixel.
    """
    max_image = None
    winner = None
    for index, (snr, threshold) in enumerate(zip(snr_images, thresholds)):
        scaled = snr / threshold
        if max_image is None:
            max_image = scaled.copy()
            winner = np.zeros(scaled.shape, dtype=np.uint8)
        else:
            better = scaled > max_image
            max_image[better] = scaled[better]
            winner[better] = index
    return max_image, winner


def _assign_segments(deblended, max_image, winner, snr_images, snr_rms,
                     footprints, shape, npixels, mask=None,
                     dilate=_SEGMENT_DILATE, max_sources=0):
    """
    Turn deblended catchments into the final segmentation image.

    Sources are painted brightest first so that a dilated segment cannot take
    pixels from a more significant neighbour.  Each source's shape comes from
    the detection image that actually fits it: the catchment intersected with
    the winning template's own footprint, so a star keeps the compact PSF
    isophote instead of the inflated one a wide template would draw.

    Returns
    -------
    segment_img : `SegmentationImage` or None
    template_index : 1D `numpy.ndarray`
    significance : 1D `numpy.ndarray`
    """
    labels = np.asarray(deblended.data)
    structure = np.ones((3, 3), dtype=bool)

    # significance in units of each SNR image's own noise, so that one number
    # means the same thing for every template
    candidates = []
    for label, slices in enumerate(scipy.ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        inside = labels[slices] == label
        peak = np.unravel_index(
            np.argmax(np.where(inside, max_image[slices], -np.inf)), inside.shape
        )
        ypeak = peak[0] + slices[0].start
        xpeak = peak[1] + slices[1].start
        index = int(winner[ypeak, xpeak])
        rms = float(snr_rms[index][ypeak, xpeak])
        sig = float(snr_images[index][ypeak, xpeak]) / (rms if rms > 0 else 1.0)
        candidates.append((sig, slices, inside, ypeak, xpeak, index))

    candidates.sort(key=lambda item: -item[0])

    merged = np.zeros(shape, dtype=np.int32)
    template_index = []
    significance = []
    n_label = 0
    n_dropped = 0
    n_trimmed = 0

    n_capped = 0
    for sig, slices, inside, ypeak, xpeak, index in candidates:
        if max_sources and n_label >= max_sources:
            # A very crowded field can yield far more sources than the
            # photometry that follows can measure in reasonable time.  The
            # candidates are ordered by significance, so stopping here keeps
            # the most significant ones and skips the rest before they are
            # painted or measured.
            n_capped = len(candidates) - max_sources
            break
        pad = int(dilate) + 1
        y0 = max(slices[0].start - pad, 0)
        y1 = min(slices[0].stop + pad, shape[0])
        x0 = max(slices[1].start - pad, 0)
        x1 = min(slices[1].stop + pad, shape[1])
        window = (slice(y0, y1), slice(x0, x1))

        local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local[
            slices[0].start - y0 : slices[0].stop - y0,
            slices[1].start - x0 : slices[1].stop - x0,
        ] = inside

        narrowed = local & footprints[index][window]
        if narrowed.sum() >= npixels:
            local = narrowed

        if dilate > 0:
            local = scipy.ndimage.binary_dilation(
                local, structure=structure, iterations=int(dilate)
            )
        local &= merged[window] == 0

        if local.any():
            # keep only the piece containing the peak; intersecting with a
            # template footprint, or losing pixels to a brighter neighbour,
            # can split a catchment into islands, and a label whose pixels
            # lie in two places has its centroid in the gap between them
            pieces, n_pieces = scipy.ndimage.label(local, structure=structure)
            if n_pieces > 1:
                keep = int(pieces[ypeak - y0, xpeak - x0])
                if keep == 0:  # peak taken by a brighter neighbour
                    counts = np.bincount(pieces.ravel())
                    counts[0] = 0
                    keep = int(counts.argmax())
                local = pieces == keep
                n_trimmed += 1

        usable = local if mask is None else local & ~mask[window]
        if int(usable.sum()) < _MIN_SEGMENT_PIXELS:
            n_dropped += 1
            continue

        n_label += 1
        merged[window][local] = n_label
        template_index.append(index)
        significance.append(sig)

    log.info(
        f"Detected {n_label} sources ({n_dropped} dropped with fewer than "
        f"{_MIN_SEGMENT_PIXELS} usable pixels, {n_trimmed} trimmed to their "
        "peak component)"
    )
    if n_capped:
        log.info(
            f"Kept the {max_sources} most significant sources; skipped "
            f"{n_capped} fainter candidates (max_sources)"
        )
    if n_label == 0:
        return None, None, None

    return (
        SegmentationImage(merged),
        np.array(template_index, dtype=np.uint8),
        np.array(significance, dtype=np.float32),
    )


def make_segmentation_image_template(
    data,
    err,
    snr_threshold,
    n_pixels,
    kernel_fwhm,
    deblend=True,
    mask=None,
    bkg_boxsize=100,
    max_sources=0,
):
    """
    Detect sources with a bank of matched filters.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        Background-subtracted data.
    err : 2D `numpy.ndarray`
        Per-pixel uncertainty.
    snr_threshold : float
        Detection threshold, in units of the significance image's own noise.
        The same number for every template, so it means "this many sigma"
        regardless of source size.
    n_pixels : int
        Minimum number of connected pixels for a deblended child.
    kernel_fwhm : float
        FWHM of the PSF template, in pixels.
    deblend : bool, optional
        Whether to deblend the maximum image.
    mask : 2D `numpy.ndarray`, optional
        Boolean mask; True values are given zero weight.
    bkg_boxsize : int, optional
        Box size for estimating the noise of each significance image.
    max_sources : int, optional
        Keep at most this many sources, the most significant first.  Zero
        means no limit.

    Returns
    -------
    segment_img : `SegmentationImage` or None
        The segmentation image.
    conv_psf : 2D `numpy.ndarray`
        The PSF-convolved flux image, for centroids and moments.
    template_index : 1D `numpy.ndarray` or None
        Index of the template that detected each source.
    significance : 1D `numpy.ndarray` or None
        Peak significance of each source, in sigma.
    """
    snr_images, conv_psf = make_template_snr_images(
        data, err, kernel_fwhm, mask=mask
    )

    thresholds = []
    snr_rms = []
    footprints = []
    for snr in snr_images:
        # ``mask``, not ``coverage_mask``: the significance is defined at a
        # masked pixel, because only the weights *under* the kernel need be
        # non-zero.  Passing the mask as coverage would blank the noise
        # estimate exactly where sources are brightest -- most masked pixels
        # are saturated cores -- leaving the threshold undefined there.
        bkg_rms = RomanBackground(
            snr, box_size=bkg_boxsize, mask=mask
        ).background_rms
        snr_rms.append(bkg_rms)
        thresholds.append(snr_threshold * bkg_rms)
        # only the footprint is used, so there is nothing to deblend here
        segm = make_segmentation_image(
            snr, snr_threshold, 1, bkg_rms, deblend=False, mask=None
        )
        footprints.append(
            np.zeros(data.shape, dtype=bool)
            if segm is None
            else np.asarray(segm.data) > 0
        )

    union = np.zeros(data.shape, dtype=bool)
    for footprint in footprints:
        union |= footprint
    if not union.any():
        return None, conv_psf, None, None

    max_image, winner = _max_detection_image(snr_images, thresholds)
    finite = np.where(np.isfinite(max_image), max_image, 0.0)

    parents, n_parents = scipy.ndimage.label(union, structure=np.ones((3, 3)))
    segm_all = SegmentationImage(parents)
    if deblend:
        deblended = deblend_sources(
            finite,
            segm_all,
            n_pixels,
            contrast=_DEBLEND_CONTRAST,
            progress_bar=False,
        )
    else:
        deblended = segm_all
    log.info(
        f"{n_parents} connected components deblended into "
        f"{deblended.n_labels} sources"
    )

    segment_img, template_index, significance = _assign_segments(
        deblended,
        finite,
        winner,
        snr_images,
        snr_rms,
        footprints,
        data.shape,
        n_pixels,
        mask=mask,
        max_sources=max_sources,
    )

    return segment_img, conv_psf, template_index, significance
