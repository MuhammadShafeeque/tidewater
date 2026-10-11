"""The exceptions of tidewater as those of OGGM."""
import functools
from contextlib import ContextDecorator

from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

from tidewater import config as _config

_ERRORS = ((_config.InvalidParamsError, InvalidParamsError),
           (_config.InvalidWorkflowError, InvalidWorkflowError))
_TIDEWATER_ERRORS = tuple(e for e, _ in _ERRORS)


def _oggm_error(err, prefix=''):
    for theirs, ours in _ERRORS:
        if isinstance(err, theirs):
            return ours(f'{prefix}{err}')
    return err


class oggm_errors(ContextDecorator):
    """Re-raise the exceptions of tidewater as those of OGGM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`, optional
        its id is put in front of the message
    """

    def __init__(self, gdir=None):
        self.prefix = '' if gdir is None else f'({gdir.rgi_id}) '

    def __enter__(self):
        return self

    def __exit__(self, etype, err, tb):
        if isinstance(err, _TIDEWATER_ERRORS):
            raise _oggm_error(err, self.prefix) from err
        return False


def translated(func):
    """:class:`oggm_errors` as a plain decorator, for hot code paths."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except _TIDEWATER_ERRORS as err:
            raise _oggm_error(err) from err
    return wrapper
