"""Scratch: exercise the forced_photometry (time domain) path."""

from astropy.table import Table

from romancal.source_catalog.source_catalog_step import SourceCatalogStep
from romancal.source_catalog.tests.test_source_catalog import (  # noqa: F401
    image_model,
)


def test_forced_photometry(image_model, function_jail):  # noqa: F811
    SourceCatalogStep.call(
        image_model, bkg_boxsize=50, kernel_fwhm=2.0, snr_threshold=5,
        npixels=10, save_results=True, output_file="source_cat.asdf",
    )
    src = Table.read("source_cat.parquet")
    src[["label", "x_centroid", "y_centroid"]].write("src.ecsv")
    SourceCatalogStep.call(
        image_model, bkg_boxsize=50, kernel_fwhm=2.0, snr_threshold=5,
        npixels=10, save_results=True, output_file="td_cat.parquet",
        forced_photometry="src.ecsv",
    )
    cat = Table.read("td_cat.parquet")
    for name in ("x", "y", "forced_psf_flags", "forced_psf_gof"):
        assert name in cat.colnames
