from warnings import warn

warn(f"Deprecated location: These modules are now maintained under the (pip installable) library named: tec")

from tec import (
    doctest_utils,
    findimports,
    import_counting,
    modules,
    packages,
    peek,
    pip_packaging,
    pip_requirements,
)
