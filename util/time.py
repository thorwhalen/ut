from warnings import warn

warn(
    'Deprecated: Moved to ut.util.utime (to avoid name conflict with standard lib `time`)'
)

from ut.util.utime import *  # move here
