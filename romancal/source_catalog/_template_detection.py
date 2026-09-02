"""
Module to detect sources with a bank of matched filters.

The image is convolved with several templates spanning a range of source
sizes.  Each convolution is turned into a significance image, the
maximum-likelihood amplitude of that template divided by its uncertainty,
and the per-pixel maximum over templates becomes a single detection image on
which sources are found via watershed and deblended.  This enables
the deblending and sounce detection to happen on a single image, reducing
challenges surrounding multiply detecting the same sources with different kernels.

However, for segment definition purposes we want to compute segments going
out to some rough isophotal limit that's appropriate for the source in question;
we do not want to convolve a bright star with a large galaxy template and
construct a segment that goes out until that large galaxy cross-correlation
has fallen off significantly.  Because each source's peak in the significance
image corresponds to a peak from a specific template, we can use the max
significance image to define a parent set of segments and then intersect
that with the specific segment that is specific for the kind of template
that generated the peak.  So PSFs end up going out to isophotes appropriate
for PSFs and large galaxies go deeper.

Additionally we use 'lowered' templates that sum to zero to prevent
small kernels from firing when they ride on a background of a large source.
"""

import logging

import numpy as np
import scipy.ndimage
from photutils.segmentation import SegmentationImage, deblend_sources

from romancal.source_catalog._background import RomanBackground
from romancal.source_catalog._detection import (
    ivw_convolve,
    make_gaussian_kernel,
    snr_from_ivw,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Template bank.  The first entry is the PSF and is sized by the step's
# ``kernel_fwhm``; the rest are exponential discs of half-light radius 4, 16
# and 64 pixels, converted to the equivalent Gaussian FWHM (FWHM ~ 1.678 r_e).
# The spacing is logarithmic so that a source of any size is within a factor
# of two of some template.
_TEMPLATE_RE = (4.0, 16.0, 64.0)  # exponential half-light radii, pixels
_RE_TO_FWHM = 1.678

# Kernel box size, in units of the template FWHM.
_SIZE_FACTOR = 4

# Background scale for the lowered templates, in units of the template FWHM.
# These effectively subtract background light on a particular scale,
# which needs to be significantly larger than the template in order to not
# significantly reduce the template's SNR, but small enough to subtract out
# potentially contaminating light from larger sources, to avoid detecting
# noise fluctuations in the wings of extended sources.
# At lower_scale = 10 the typical SNRs are down by a couple percent
# compared with unlowered templates.
_LOWER_SCALE = 10.0

# Deblending contrast for the maximum image: the fraction of a parent's flux a
# peak must hold to be split off.  Note this differs from the 1e-4 that
# ``make_segmentation_image`` uses for its own deblending.
_DEBLEND_CONTRAST = 0.001

# Grow each final segment by this many pixels.  We are willing to detect
# sources that are 5 sigma in a single pixel, but want to then grow those segments
# to be at least large enough to compute a moment.  We dilate so that isolated
# single pixels grow to 3x3 segments.
_SEGMENT_DILATE = 1

# Smallest final segment to keep, counted in pixels the photometry can
# actually use.  Dilation grows a lone pixel to a full 3x3, so for faint isolated
# sources this means the source must retain its 3x3 neighborhood after
# removing masked pixels and pixels assigned to neighbors.
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
        The PSF-convolved flux image, for centroids and moments.
        Note that the snr_images use a lowered kernel while
        this uses the corresponding unlowered kernel.
    """
    good = ~mask if mask is not None else np.ones(data.shape, dtype=bool)
    wht = np.where(good, 1.0 / np.where(good, err, 1.0) ** 2, 0.0)

    snr_images = []
    for fwhm in _template_fwhms(kernel_fwhm):
        kernel = make_gaussian_kernel(
            fwhm, lower_scale=_LOWER_SCALE, size_factor=_SIZE_FACTOR
        )
        if kernel.shape[0] > min(data.shape):
            # kernel plus its background-subtraction surround is wider than
            # the image, so the kernel is all boundary.  Triggered by cutouts
            # and by test frames; real images are far larger than the bank.
            log.info(
                f"Skipping template FWHM={fwhm:.1f} px: kernel "
                f"{kernel.shape[0]} px exceeds the image"
            )
            continue
        num, denom2, _ = ivw_convolve(data, wht, kernel, mask=mask)
        snr_images.append(snr_from_ivw(num, denom2))
        log.info(f"Template FWHM={fwhm:.1f} px: kernel {kernel.shape[0]} px")

    if not snr_images:
        log.warning(
            f"No template fits within an image of shape {data.shape}; "
            "no sources can be detected"
        )
        return [], None

    psf_kernel = make_gaussian_kernel(kernel_fwhm, size_factor=_SIZE_FACTOR)
    _, _, conv_psf = ivw_convolve(data, wht, psf_kernel, mask=mask)

    return snr_images, conv_psf


def _max_detection_image(snr_images, thresholds):
    """
    Per-pixel maximum over templates, after dividing by template-dependent threshold.

    Thresholds can account for the fact that usually larger templates have
    SNR images that are larger than 1, due to leakage of signal from sources into
    the background images and potentially other sources of correlated noise not
    accounted by the uncertainty images.

    Returns
    -------
    max_image : 2D `numpy.ndarray`
        The maximum image.
    template : 2D `numpy.ndarray`
        Index of the template attaining the maximum at each pixel.
    """
    max_image = None
    template = None
    for index, (snr, threshold) in enumerate(zip(snr_images, thresholds)):
        scaled = snr / threshold
        if max_image is None:
            max_image = scaled.copy()
            template = np.zeros(scaled.shape, dtype=np.uint8)
        else:
            better = scaled > max_image
            max_image[better] = scaled[better]
            template[better] = index
    return max_image, template


def _assign_segments(deblended, max_image, template_at_pixel, snr_images,
                     snr_rms,
                     footprints, shape, npixels, mask=None,
                     dilate=_SEGMENT_DILATE, max_sources=0):
    """
    Turn deblended segments into the final segmentation image.

    Parameters
    ----------
    deblended : `SegmentationImage`
        The deblended catchments, one label per source, covering the whole
        detection footprint.
    max_image : 2D `numpy.ndarray`
        The maximum-over-templates image, in units of each template's own
        threshold.  Used only to locate each catchment's peak.
    template_at_pixel : 2D `numpy.ndarray`
        Index of the template attaining the maximum at each pixel.
    snr_images : list of 2D `numpy.ndarray`
        The per-template SNR images, used to read each source's peak value.
    snr_rms : list of 2D `numpy.ndarray`
        The measured background noise of each SNR image, so that the peak can
        be expressed in sigma.
    footprints : list of 2D `numpy.ndarray`
        Boolean footprint of each template, i.e. where that template's SNR
        exceeds its own threshold.  Used to narrow a catchment to the
        isophote of the template that fits it.
    shape : tuple
        Shape of the output segmentation image.
    npixels : int
        A catchment is narrowed to its template's footprint only if at least
        this many pixels survive; otherwise the whole catchment is kept.
    mask : 2D `numpy.ndarray`, optional
        Boolean mask.  Masked pixels stay inside a segment, so that a bad
        column does not cut a source in two, but they do not count toward
        the minimum size, because the photometry cannot use them.
    dilate : int, optional
        Grow each segment by this many pixels before painting.
    max_sources : int, optional
        Stop after this many sources have been painted.  Zero means no limit.

    Returns
    -------
    segment_img : `SegmentationImage` or None
    template_index : 1D `numpy.ndarray`
    significance : 1D `numpy.ndarray`

    Notes
    -----
    Sources are painted brightest first so that a dilated segment cannot take
    pixels from a more significant neighbour.  Each source's shape comes from
    the detection image that actually fits it: the segment intersected with
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
        index = int(template_at_pixel[ypeak, xpeak])
        rms = float(snr_rms[index][ypeak, xpeak])
        sig = float(snr_images[index][ypeak, xpeak]) / (rms if rms > 0 else 1.0)
        candidates.append((sig, slices, inside, ypeak, xpeak, index))

    candidates.sort(key=lambda item: -item[0])

    merged = np.zeros(shape, dtype=np.int32)
    # ``claimed`` tracks every pixel a source has taken, masked or not, so a
    # later source cannot dilate into a brighter one's masked core.  Only the
    # unmasked pixels are painted into the segmentation image itself.
    claimed = np.zeros(shape, dtype=bool)
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
        local &= ~claimed[window]

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

        # Connectivity is judged on the mask-inclusive footprint above, so a
        # masked stripe through a source does not split it; membership is the
        # unmasked pixels, so the segmentation image never claims a pixel the
        # photometry cannot use.
        usable = local if mask is None else local & ~mask[window]
        if int(usable.sum()) < _MIN_SEGMENT_PIXELS:
            n_dropped += 1
            continue

        n_label += 1
        claimed[window] |= local
        merged[window][usable] = n_label
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
        # mask rather than coverage_mask because snr_image is well defined
        # on masked pixels, and using mask leads bkg_rms to be well defined
        # there
        bkg_rms = RomanBackground(
            snr, box_size=bkg_boxsize, mask=mask
        ).background_rms
        snr_rms.append(bkg_rms)
        thresholds.append(snr_threshold * bkg_rms)
        footprints.append(snr > snr_threshold * bkg_rms)

    max_image, template_at_pixel = _max_detection_image(snr_images, thresholds)
    finite = np.where(np.isfinite(max_image), max_image, 0.0)

    # The detection footprint is simply where some template exceeds its own
    # threshold, which is where the maximum image exceeds one.  Its connected
    # components are the regions that will be deblended; ``deblend_sources``
    # subdivides the labels of a segmentation image, so they have to be
    # labelled first.
    union = finite > 1.0
    if not union.any():
        return None, conv_psf, None, None

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
        template_at_pixel,
        snr_images,
        snr_rms,
        footprints,
        data.shape,
        n_pixels,
        mask=mask,
        max_sources=max_sources,
    )

    return segment_img, conv_psf, template_index, significance
