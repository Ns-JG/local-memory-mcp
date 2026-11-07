class LocalMemoryError(Exception):
    """Base for every error this package raises."""


class NotFoundError(LocalMemoryError):
    """No record with the requested id."""


class ConflictError(LocalMemoryError):
    """A concurrent writer changed the record after it was read.

    Raised instead of silently overwriting, so the caller re-reads and retries.
    """

    def __init__(self, record_id: str) -> None:
        super().__init__(f"record {record_id} was modified concurrently; re-read it and retry")
        self.record_id = record_id


class ValidationError(LocalMemoryError):
    """A field failed validation on write."""


class ConfigError(LocalMemoryError):
    """config.toml is missing, unparseable or failed validation."""
