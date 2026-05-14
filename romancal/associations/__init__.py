"""Roman associations

Public API: load_asn, AssociationError, AssociationNotValidError.

The Roman-specific mosaic tools (skycell_asn, multiband_asn,
mk_skycell_list, mk_skycell_asn_from_skycell_list, asn_from_list) are
importable directly from their submodules.

The remaining infrastructure (association generation engine, pool, registry,
rules) is internal and will be removed in a future release.
"""

# flake8: noqa: F401

from .. import __version__


def libpath(filepath):
    """Return the full path to the module library."""
    from os.path import abspath, dirname, join

    return join(dirname(abspath(__file__)), "lib", filepath)


from . import _association_io  # registers JSON/YAML I/O handlers as a side effect
from .exceptions import AssociationError, AssociationNotValidError
from .load_asn import load_asn
