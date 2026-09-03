"""
Tests for the matched-filter template detection.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D

from romancal.source_catalog._background import RomanBackground
from romancal.source_catalog._detection import (
    ivw_convolve,
    make_gaussian_kernel,
    snr_from_ivw,
)
from romancal.source_catalog._template_detection import (
    _LOWER_SCALE,
    _SIZE_FACTOR,
    _template_fwhms,
    make_segmentation_image_template,
)

@pytest.fixture
def wide_image():
    """A frame large enough for the whole template bank, with sources
    spanning the range of template sizes plus a close pair."""
    shape = (700, 700)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    sources = (
        Gaussian2D(300.0, 150, 150, 0.85, 0.85),  # point source
        Gaussian2D(60.0, 350, 150, 3.0, 3.0),  # small galaxy
        Gaussian2D(12.0, 150, 380, 12.0, 12.0),  # large galaxy
        Gaussian2D(200.0, 520, 500, 0.85, 0.85),  # close pair, bright
        Gaussian2D(200.0, 536, 500, 0.85, 0.85),  # close pair, companion
    )
    data = sum(src(xx, yy) for src in sources).astype("float32")
    rng = np.random.default_rng(seed=42)
    data += rng.normal(0.0, 1.0, size=shape)
    err = np.ones(shape, dtype=np.float32)
    return data.astype("float32"), err


def _detect(data, err, mask=None, **kwargs):
    bkg = RomanBackground(data, box_size=100, mask=mask)
    params = dict(
        snr_threshold=5.0, n_pixels=1, kernel_fwhm=2.0, deblend=True,
        mask=mask, bkg_boxsize=100,
    )
    params.update(kwargs)
    return make_segmentation_image_template(data - bkg.background, err, **params)


def test_lowered_kernel_is_zero_sum():
    """A lowered kernel gives no response to a flat background."""
    plain = make_gaussian_kernel(2.0, size_factor=_SIZE_FACTOR)
    lowered = make_gaussian_kernel(
        2.0, lower_scale=_LOWER_SCALE, size_factor=_SIZE_FACTOR
    )
    assert np.isclose(plain.sum(), 1.0)
    assert abs(lowered.sum()) < 1e-8
    assert lowered.shape[0] > plain.shape[0]  # room for the background subtraction kernel
    # the SNR loss is small
    assert np.sqrt((lowered**2).sum()) / np.sqrt((plain**2).sum()) > 0.9


def test_significance_defined_on_masked_pixels():
    """The matched filter is well defined in small masked regions."""
    data = np.zeros((60, 60), dtype=np.float32)
    err = np.ones_like(data)
    mask = np.zeros(data.shape, dtype=bool)
    mask[30, 30] = True
    wht = np.where(mask, 0.0, 1.0 / err**2)
    num, denom2, _ = ivw_convolve(data, wht, make_gaussian_kernel(2.0), mask=mask)
    assert denom2[30, 30] > 0
    assert np.isfinite(snr_from_ivw(num, denom2)[30, 30])


def test_templates_larger_than_image_are_skipped(wide_image):
    """Too-large kernels for an image are skipped."""
    data, err = wide_image
    small = data[:120, :120].copy(), err[:120, :120].copy()
    segment_img, _, template_index, _ = _detect(*small)
    if segment_img is not None:
        # only the PSF template fits inside a 120 px frame
        assert np.all(template_index == 0)


def test_detects_sources_of_each_size(wide_image):
    """Sources are assigned to the template that fits them."""
    data, err = wide_image
    segment_img, conv_psf, template_index, significance = _detect(data, err)

    assert segment_img is not None
    assert conv_psf.shape == data.shape
    assert len(template_index) == segment_img.n_labels
    assert len(significance) == segment_img.n_labels
    assert np.all(significance >= 5.0)  # the threshold is the floor
    assert len(_template_fwhms(2.0)) == 4

    # the point source and the large galaxy are not assigned the same template
    labels = np.asarray(segment_img.data)
    point = template_index[labels[150, 150] - 1]
    extended = template_index[labels[380, 150] - 1]
    assert point < extended


def test_segments_are_contiguous(wide_image):
    """Every label is a single connected piece.

    Not automatic: narrowing a max-image segment to its template's footprint, and
    losing pixels to a brighter neighbour, can both split it into islands,
    and a label whose pixels lie in two places has its centroid in the gap
    between them.  `_assign_segments` trims each source to the piece holding
    its peak, and this guards that.
    """
    import scipy.ndimage

    data, err = wide_image
    segment_img, _, _, _ = _detect(data, err)
    labels = np.asarray(segment_img.data)
    structure = np.ones((3, 3), dtype=bool)
    for label in np.unique(labels[labels > 0]):
        _, n_pieces = scipy.ndimage.label(labels == label, structure=structure)
        assert n_pieces == 1


def test_close_pair_is_deblended(wide_image):
    """Two point sources 16 px apart become two sources."""
    data, err = wide_image
    segment_img, _, _, _ = _detect(data, err)
    labels = np.asarray(segment_img.data)
    assert labels[500, 520] != labels[500, 536]
    assert labels[500, 520] > 0 and labels[500, 536] > 0


def test_masked_stripe_does_not_split_a_segment(wide_image):
    """A masked stripe through a source -- a bleed trail, say -- must not cut
    its segment in two.  The significance and the detection threshold are both
    defined on masked pixels, so the segment closes over the stripe."""
    import scipy.ndimage

    data, err = wide_image
    mask = np.zeros(data.shape, dtype=bool)
    mask[375:386, 150] = True  # a stripe across the large galaxy

    segment_img, _, _, _ = _detect(data, err, mask=mask)
    assert segment_img is not None
    labels = np.asarray(segment_img.data)

    # the masked pixels themselves are not claimed by any source ...
    assert labels[378, 150] == 0
    # ... but the source is one label, not two, on either side of the stripe
    above, below = labels[372, 150], labels[389, 150]
    assert above > 0
    assert above == below


def test_size_cut_counts_moment_positive_pixels(wide_image, monkeypatch):
    """The minimum segment size counts the pixels the moments can weight.

    Photutils drops non-positive pixels before taking moments, so a segment
    with none of them has no centroid and returns NaN, which later stops the
    step at the PSF fit.  With a moment image that is positive nowhere, every
    source must be dropped rather than kept with an undefined centroid.
    """
    import romancal.source_catalog._template_detection as td

    real = td.make_template_snr_images

    def negated(*args, **kwargs):
        snr_images, conv = real(*args, **kwargs)
        return snr_images, -np.abs(conv)

    monkeypatch.setattr(td, "make_template_snr_images", negated)
    segment_img, _, _, _ = _detect(*wide_image)
    assert segment_img is None
