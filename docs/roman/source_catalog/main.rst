Description
===========

:Class: `romancal.source_catalog.SourceCatalogStep`
:Alias: source_catalog

This step detects sources in an image and generates a catalog of their
properties, including photometric and shape measurements.


Background Subtraction
----------------------

A two-dimensional background is estimated and subtracted from the
data. The background and background noise are estimated using the
:external+photutils:py:class:`photutils.background.Background2D`
class from `Photutils
<https://photutils.readthedocs.io/en/stable/index.html>`_. This class
calculates the background by measuring the sigma-clipped median within
user-defined boxes of a specified size (``bkg_boxsize``). The background
RMS noise is then estimated using the sigma-clipped standard deviation
within the same boxes.


Source Detection
----------------

Sources are detected using `image segmentation
<https://en.wikipedia.org/wiki/Image_segmentation>`_, a process that
assigns an integer label to each pixel in an image, such that pixels
with the same label correspond to the same source. Source extraction is
then performed using the `Photutils segmentation <https://photutils.readthedocs.io/en/latest/user_guide/segmentation.html>`_ tools.

The background-subtracted image is filtered with a bank of matched
filters rather than a single kernel, so that sources of different sizes
are each detected by a template close to their own scale. The bank holds
a point-source template, a Gaussian whose FWHM is set by the
``kernel_fwhm`` parameter, together with three larger templates standing
in for galaxies, with half-light radii of 4, 16, and 64 pixels. Each
filter is applied with inverse-variance weighting, so what it produces
is the maximum-likelihood amplitude of that template divided by its own
uncertainty: a significance image, in units of sigma. Masked pixels
enter with zero weight rather than being excluded, so the significance
is still defined where individual pixels are bad, and a bad column no
longer punches a hole through the middle of a source.

Every template is *lowered*: the mean over the kernel box is subtracted
so that the template sums to zero. A zero-sum filter gives no response
to a flat background, so a compact source sitting on the light of a
larger galaxy is measured by how far it stands above that light rather
than by the total. The box is large compared with the template it
carries, since the scale that has to be removed is set by the
contaminating source rather than by the source being detected.

Sources are then found on the per-pixel maximum over the templates.
Because each template's image is already in units of its own noise, the
maximum is itself a significance image, recording the best evidence any
template can offer at that pixel. Pixels above ``snr_threshold`` in that
single image form the detection footprint, and overlapping sources are
separated there by the `Photutils deblender
<https://photutils.readthedocs.io/en/latest/user_guide/segmentation.html
#source-deblending>`_, which applies a multi-thresholding approach and
then `watershed segmentation
<https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. For
successful deblending, the sources must be separated enough that there
is a saddle point between them. Deblending one combined image, rather than each filter separately,
means every pixel belongs to exactly one source and no filter can claim
a parent source enclosing several others.

Each source is finally given the extent appropriate to its own kind.
The template attaining the maximum at a source's peak is the one that
fits it best, and the source's segment is narrowed to that template's
significant pixels, so a star keeps a compact isophote where a galaxy
goes out to a faint one. Segments are grown by one pixel and painted in
order of decreasing significance, so that a fainter source cannot take
pixels from a brighter neighbour, and any segment left with fewer than
``npixels`` usable pixels is dropped. The template that detected each
source and its peak significance are recorded in the catalog as
``det_template`` and ``det_significance``. Very crowded fields can yield
more detections than are worth measuring, so ``max_sources`` keeps only
that many of the most significant.


Source Photometry and Properties
--------------------------------

After detecting sources using image segmentation, we measure their
photometry, centroids, and shape/morphological properties.

The source centroids and shape properties are derived from 2D
image moments of the pixel values within the source segments using
:external+photutils:py:class:`photutils.segmentation.SourceCatalog`.
These properties include the semimajor and semiminor axes, ellipticity,
and orientation of the major axis.

Circular aperture photometry is performed at several aperture sizes
(:math:`r` = 0.1, 0.2, 0.4, 0.8, 1.6 arcsec) for each source. Elliptical
Kron aperture photometry is also performed, where the aperture size is
determined by the source shape.

Isophotal photometry is measured using the total flux within the source
segment.

Optionally, Point Spread Function (PSF) photometry can be performed
by setting the ``fit_psf`` keyword. Enabling this option fits a model
PSF to each source to measure its position and flux. The PSF model is
generated using reference files on CRDS. PSF photometry is performed
using the :external+photutils:py:class:`photutils.psf.PSFPhotometry`
class.

For Level 2 data, a gridded PSF model is generated for each individual
detector using the reference files in CRDS. These PSF models account
for jitter by deconvolving the amount of jitter indicated to be
present in the PSF reference file, and reconvolving with the amount
of jitter present in the individual image. Because the amount
of jitter present in the reference file is small (~8 mas), this
deconvolution does not introduce significant noise. The convolution
process works in the Fourier domain using the approach of `Lang (2020)
<https://ui.adsabs.harvard.edu/abs/2020arXiv201215797L/abstract>`_
because the jitter kernel would otherwise be badly undersampled.

For Level 3 data, since the data contains a mixture of individual
detector PSFs with different orientations, further processing is done.
The base PSF is calculated for the center of the WFI02 detector. It
is then scaled and smoothed to roughly account for the different
pixel scale of the coadded images relative to the detector images,
and the effect of the image drizzling on the PSF. Finally, the PSF is
azimuthally averaged to remove any azimuthal signatures, which will be
different in the coadded product than in the individual input exposures.

All fluxes are reported in nJy. To calculate AB magnitudes, use the
following formula:

.. math::

    m_{\rm AB} = -2.5 \log_{10}(f_{\rm nJy}) + 31.4

Photometric errors are calculated from the resampled total-error array
contained in the ``model.err`` array. Note that this array includes
source Poisson noise.

A local background is estimated for each source measured within a
circular annulus centered on the source. The circular annulus has an
inner and outer radius of 2.4 and 2.8 arcsec, respectively. The local
background flux is calculated as the sigma-clipped median value within
the annulus. Although this local background value is included in the
source catalog, it is not subtracted from any of the measured fluxes.

Each source has a field, `is_extended`, intended to indicate whether the
source is more extended than expected, were the object a point source.
The determination is made on the basis of the ratio of aperture fluxes
at 0.4 and 0.2 arcsec being more than 1.2x larger than expected for
a point source.


Source Catalog Table
--------------------

The source catalog table contains one row for each source, with the
columns listed below (assuming PSF-photometry is requested, i.e.,
``fit_psf=True``).

All pixel coordinates are 0-indexed, following Python's 0-based
indexing. This means pixel coordinate 0 corresponds to the center of the
first pixel.

All sky coordinates are in decimal degrees in the International
Celestial Reference System (ICRS) reference frame.

Uncertainties are reported as the 1-sigma (68.27% confidence) errors.

Some column names contain templated strings that will be replaced with
values specific to the generated file. For example ``~band~`` will be
replaced with a filter wavelength band (for example ``f184``) where
appropriate and removed for single-filter files. ``~radius~`` will
be replaced with the aperture radius in tenths of an arcsecond. For
example, a single filter catalog with 0.1 arcsecond aperture photometry
will contain an ``aper01_flux`` column. A catalog derived from multiple
filters (including ``f184``) and the same aperture radius will contain
an ``aper01_f184_flux`` column.

.. source_catalog_columns::

Further details for some of the columns are provided below.

``flagged_spatial_id`` is a bit flag encoding the overlap flag,
projection, skycell, and pixel coordinates of the source. Bit positions
are 0-indexed (i.e., bit N has value ``2**N``). From high to low, bit
59 is 1 if the object was outside of the core region of this skycell or
projection region. There is likely to be a better measurement of the
object in a different skycell with this bit set to 0. This bit is the
same as bit **TBD** of ``warning_flags``. Bits 46-58 encode the primary
projection region for this object. Bits 32-38 and 39-45 encode the x and
y skycell indices within this projection region, starting from (0, 0) at
the lower left. Bits 0-15 and 16-31 encode the x and y pixel coordinates
of the object within this skycell in virtual 0.05" pixels (regardless of
the pixel scale of the skycell).

The ``psf_gof`` metric is the reduced chi-squared of the PSF fit.


Flag Columns
^^^^^^^^^^^^

The ``warning_flags`` column contains the following bit flags:

- 0 : Good
- 1 :

  * Level 2: sources whose rounded centroid pixel is not finite or has
    DO_NOT_USE set in the model DQ

  * Level 3: sources whose rounded centroid pixel is not finite or has a
    weight of 0

The ``image_flags`` column contains the following bit flags:

- 0 : Good
- 1 : One or more pixels in the source segment was flagged

The ``psf_flags`` column contains the following bit flags defined by the
:external+photutils:py:class:`photutils.psf.PSFPhotometry` class:

- 0 : Good
- 1 : One or more pixels in the ``fit_shape`` region were masked
- 2 : The fitted source position is outside the bounds of the input image
- 4 : The fitted flux value is negative or zero
- 8 : The PSF fitting algorithm may not have converged to a stable solution
- 16 : Parameter covariance matrix is not available, preventing error estimation
- 32 : One or more fitted parameters are very close to their imposed bounds
- 64 : The source PSF fitting region has no overlap with valid data pixels
- 128 : All pixels in the source fitting region are masked
- 256 : Insufficient unmasked pixels available for reliable PSF fitting
- 512 : The fitted x or y position is NaN or inf, indicating an invalid or failed fit
- 1024 : The fitted flux value is NaN or inf, indicating an invalid or failed fit
- 2048 : The local background value is NaN or inf, so it was not subtracted before fitting


Output Products
---------------

Source Catalog Table
^^^^^^^^^^^^^^^^^^^^

The output source catalog table is saved to a file in the `Parquet
<https://parquet.apache.org/>`_ format.


Segmentation Map
^^^^^^^^^^^^^^^^

The segmentation map generated during the
source-finding process is saved as an `ASDF
<https://en.wikipedia.org/wiki/Advanced_Scientific_Data_Format>`_ file.
Each pixel in the image contains an integer value corresponding to a
source label in the source catalog. Pixels that do not belong to any
source are assigned a value of zero.

For L2 imaging products, when ``compute_skyvals=True`` (default), the
segmentation output also includes ``skyvals`` and ``healpix11_cov``
arrays containing HEALPix-based sky summary values for downstream
sky-matching workflows.


Multiband Catalogs
------------------

Multiband catalogs combine multiple images to create a deep detection
image, which is used to detect sources and identify segments. The
measured positions and shapes of the sources in these deep images are
then used to perform aperture, Kron, isophotal, and PSF photometry for
each filter.

The catalog fields are similar to those in the source catalog schema,
but with the following differences:

* Fields derived from individual filter images include the
  filter name from which they were derived. For example, fields
  like ``aper_flux_<filter>``, ``segment_flux_<filter>``,
  ``kron_flux_<filter>``, and ``psf_flux_<filter>`` provide the aperture
  and PSF flux for each filter, respectively.

* Fields derived from the detection image and segmentation map do not
  include the filter name.

Multiband catalogs are generated by the
:py:class:`~romancal.multiband_catalog.MultibandCatalogStep`, which
takes an association file as input. This file lists the images that need
to be photometered simultaneously.


Forced Source Catalogs
----------------------

Source catalogs can optionally be generated by using the segmentation
image from one image (the "forcing" image) and computing shapes and
fluxes for those same segments in another image (the "forced" image).
For this to work, the two images must be perfectly aligned in pixel
space.

Forced source catalogs can be generated by specifying a segmentation
image with the ``forced_segmentation`` keyword when running the source
catalog step.

In this mode, the source catalog contains fields with the ``forced``
prefix, in addition to the fields described above. Fields without the
"forced" prefix contain position and shape information derived from
the forcing image, indicating where measurements were taken on the
forced image. Fields with the forced prefix represent values computed
on the forced image, using the information from the forcing image.
For example, the field ``forced_kron_flux`` represents the Kron flux
measured on the forced image, using the centroid and shape information
from the ``x_centroid``, ``y_centroid``, ``semimajor``, ``semiminor``,
and ``orientation_pix`` fields.
