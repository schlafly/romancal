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
from photutils.segmentation import SegmentationImage

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
_LOWER_SCALE = 12.0

# How the background is subtracted: see ``make_gaussian_kernel``.
_LOWER_MODE = "box"

# Whether the image used for centroids and moments is also lowered.
#
# Lowering it is measurably better where it works: on the galaxy field the
# median centroid error against truth falls from 0.684 to 0.626 px, and on the
# star field the measured source size becomes 0.13 arcsec against 0.21, the
# former matching the expected convolved PSF width of 1.20 px -- so the
# unlowered moments are biased by the background pedestal they still contain.
#
# It is nevertheless off, because it fails on crowded fields: this one image is
# the PSF-scale convolution for every segment, whichever template detected it,
# and where the local mean is dominated by crowd or galaxy light the lowered
# image can be negative over a whole footprint.  Moments then have no positive
# pixel to weight and come back as NaN.
_MOMENT_LOWERED = False

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
            fwhm, lower_scale=_LOWER_SCALE, size_factor=_SIZE_FACTOR,
            lower_mode=_LOWER_MODE,
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

    psf_kernel = make_gaussian_kernel(
        kernel_fwhm,
        size_factor=_SIZE_FACTOR,
        lower_scale=_LOWER_SCALE if _MOMENT_LOWERED else 0.0,
        lower_mode=_LOWER_MODE,
    )
    _, _, conv_psf = ivw_convolve(data, wht, psf_kernel, mask=mask)
    if _MOMENT_LOWERED:
        # Moments weight segment pixels, so the weights must be non-negative
        # and a lowered kernel's negative surround breaks that.  Photutils
        # drops negative pixels itself, so this is belt and braces, and it is
        # not sufficient; see the note beside _MOMENT_LOWERED.
        conv_psf = np.maximum(conv_psf, 0.0)

    return snr_images, conv_psf


def _max_detection_image(snr_images):
    """
    Per-pixel maximum over the templates' significance images.

    Each input is already in units of its own background noise, so they are
    directly comparable and the maximum is itself a significance image: the
    best evidence any template can offer at that pixel.

    Parameters
    ----------
    snr_images : list of 2D `numpy.ndarray`
        One significance image per template, in sigma.

    Returns
    -------
    max_image : 2D `numpy.ndarray`
        The per-pixel maximum, in sigma.
    template_at_pixel : 2D `numpy.ndarray`
        Index of the template attaining the maximum at each pixel.
    """
    max_image = None
    template_at_pixel = None
    for index, snr in enumerate(snr_images):
        if max_image is None:
            max_image = snr.copy()
            template_at_pixel = np.zeros(snr.shape, dtype=np.uint8)
        else:
            better = snr > max_image
            max_image[better] = snr[better]
            template_at_pixel[better] = index
    return max_image, template_at_pixel


def _assign_segments(deblended, max_image, template_at_pixel, footprints,
                     shape, mask=None, dilate=_SEGMENT_DILATE, max_sources=0):
    """
    Turn deblended segments into the final segmentation image.

    Start with a set of segments derived from the max_image, and refine
    those by looping over segments in order of their peak SNR.  For each
    segment, find the template that corresponds to it, and extract the
    corresponding segment derived from that SNR image.  Intersect it
    with the max_image SNR image and paint it onto final derived
    segmentation image, after dilating.  Later lower SNR segments
    cannot overwrite pixels claimed by an earlier segment, and are dropped
    if they end up with fewer than _MIN_SEGMENT_PIXELS valid pixels.

    Parameters
    ----------
    deblended : `SegmentationImage`
        The deblended segments from the max_image, one label per source,
        covering the whole detection footprint.
    max_image : 2D `numpy.ndarray`
        The maximum-over-templates significance image, in sigma.  Used to
        locate each segment's peak and to read its significance there.
    template_at_pixel : 2D `numpy.ndarray`
        Index of the template attaining the maximum at each pixel.
    footprints : list of 2D `numpy.ndarray`
        Boolean footprint of each template, i.e. where that template's SNR
        exceeds its own threshold.  Used to narrow segments from max_det to
        the segments from the template that fits it.
    shape : tuple
        Shape of the output segmentation image.
    mask : 2D `numpy.ndarray`, optional
        Boolean mask indicating bad pixels.  Masked pixels stay inside a
        segment, so that a bad column does not cut a segment into two.  However,
        masked pixels do not count towards the minimum segment size.
    dilate : int, optional
        Grow each segment by this many pixels before painting.
    max_sources : int, optional
        Stop after this many sources have been painted.  Zero means no limit.

    Returns
    -------
    segment_img : `SegmentationImage` or None
    template_index : 1D `numpy.ndarray`
    significance : 1D `numpy.ndarray`

    Returns
    -------
    segment_img : `SegmentationImage` or None
    template_index : 1D `numpy.ndarray`
    significance : 1D `numpy.ndarray`
    """
    labels = np.asarray(deblended.data)
    structure = np.ones((3, 3), dtype=bool)

    # Each segment's peak, and the template that wins there.  The winning
    # template is the argmax, so the maximum image at the peak *is* that
    # template's significance -- there is nothing further to divide by.
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
        candidates.append(
            (float(max_image[ypeak, xpeak]), slices, inside, ypeak, xpeak, index)
        )

    # brightest first
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
            # Allow early termination for crowded fields if requested, keeping
            # the bright sources.
            n_capped = len(candidates) - max_sources
            break

        # pad to allow segment dilation
        pad = int(dilate) + 1
        y0 = max(slices[0].start - pad, 0)
        y1 = min(slices[0].stop + pad, shape[0])
        x0 = max(slices[1].start - pad, 0)
        x1 = min(slices[1].stop + pad, shape[1])
        window = (slice(y0, y1), slice(x0, x1))

        # construct nominal segment in padded stamp
        local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local[
            slices[0].start - y0 : slices[0].stop - y0,
            slices[1].start - x0 : slices[1].stop - x0,
        ] = inside

        # Narrow to the isophote of the template that fits this source, so a
        # star keeps the compact PSF isophote rather than the inflated one a
        # wide template draws.  The segment supplies the isolation (segments
        # are disjoint) and the footprint supplies only the extent, so a
        # binary footprint suffices and no cross-template matching is needed.
        # The peak is always inside its own template's footprint -- winning
        # the maximum there means exceeding that template's threshold -- so
        # this can never empty the segment.
        local &= footprints[index][window]

        # dilate local segment
        if dilate > 0:
            local = scipy.ndimage.binary_dilation(
                local, structure=structure, iterations=int(dilate)
            )
        # restrict to pixels that haven't already been claimed
        local &= ~claimed[window]

        if local.any():
            # intersecting with a template footprint or losing pixels to
            # a brighter neighbour can split a segment into islands.
            # Here we trim the segment to the island containing the peak.
            pieces, n_pieces = scipy.ndimage.label(local, structure=structure)
            if n_pieces > 1:
                keep = int(pieces[ypeak - y0, xpeak - x0])
                if keep == 0:  # peak taken by a brighter neighbour
                    counts = np.bincount(pieces.ravel())
                    counts[0] = 0
                    keep = int(counts.argmax())
                # trimming to the segment containing the peak
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
        Detection threshold in sigma, the same for every template.
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

    # Put each template's SNR image in units of its own background noise, so
    # that one threshold means the same thing for every template and the
    # maximum over templates is itself a significance image in sigma.  The
    # noise is not 1 in practice: signal leaks into the background estimate,
    # and resampling correlates the noise, both more so for wider kernels.
    footprints = []
    for i, snr in enumerate(snr_images):
        # mask rather than coverage_mask because snr_image is well defined
        # on masked pixels, and using mask leads bkg_rms to be well defined
        # there
        bkg_rms = RomanBackground(
            snr, box_size=bkg_boxsize, mask=mask
        ).background_rms
        snr_images[i] = np.where(bkg_rms > 0, snr / bkg_rms, 0.0)
        footprints.append(snr_images[i] > snr_threshold)

    max_image, template_at_pixel = _max_detection_image(snr_images)
    max_image = np.where(np.isfinite(max_image), max_image, 0.0)

    # Detection and deblending on that single image, using the pipeline's own
    # segmentation helper.  A unit background RMS is passed because the image
    # is already in sigma.
    deblended = make_segmentation_image(
        max_image,
        snr_threshold,
        n_pixels,
        1.0,
        deblend=deblend,
        mask=None,
        deblend_contrast=_DEBLEND_CONTRAST,
    )
    if deblended is None:
        return None, conv_psf, None, None
    log.info(f"{deblended.n_labels} sources after deblending")

    segment_img, template_index, significance = _assign_segments(
        deblended,
        max_image,
        template_at_pixel,
        footprints,
        data.shape,
        mask=mask,
        max_sources=max_sources,
    )

    return segment_img, conv_psf, template_index, significance
