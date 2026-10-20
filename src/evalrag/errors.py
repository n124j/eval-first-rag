"""Shared runtime error types."""


class CacheMismatchError(ValueError):
    """A record cache was made with a different model, dimension or parameters."""


class IndexDataError(ValueError):
    """An on-disk index file is malformed."""
