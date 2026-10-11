"""The OGGM host of tidewater: tasks, calving laws and the settings they read.

Call :func:`register` once, before or after ``cfg.initialize()`` and before
the glacier directories are opened. The tasks are in
:mod:`tidewater.hosts.oggm.tasks`.
"""
from tidewater.hosts.oggm.params import add_basenames as _add_basenames

# The tasks declare the files they write when they are defined
_add_basenames()


def supported():
    """Whether the installed OGGM has the glacier settings and data store (later than 1.6.3)."""
    from oggm import GlacierDirectory

    return hasattr(GlacierDirectory, 'read_store')


def register():
    """Add the files, the parameters and the calving inversion of the scheme to OGGM.

    The parameters get their defaults where ``cfg.PARAMS`` lacks them, and
    every later ``cfg.initialize`` sets them again, from its ``file`` and
    ``params`` too. Multiprocessing workers receive them with the rest of
    the configuration. Calling it again changes nothing.
    """
    from oggm import workflow
    from oggm.exceptions import InvalidWorkflowError

    from tidewater.hosts.oggm.inversion import find_inversion_calving
    from tidewater.hosts.oggm.params import set_params, wrap_initialize

    if not supported():
        raise InvalidWorkflowError('tidewater.hosts.oggm needs an OGGM later '
                                   'than 1.6.3 (gdir.settings, read_store).')
    _add_basenames()
    wrap_initialize()
    set_params()
    # A hook of the OGGM branch; stock OGGM ignores it
    if getattr(workflow, 'CALVING_INVERSION_TASK', None) is None:
        workflow.CALVING_INVERSION_TASK = find_inversion_calving
