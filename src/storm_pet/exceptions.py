class StormPetError(RuntimeError):
    """Base exception for the STORM pipeline."""


class ConfigurationError(StormPetError):
    """Raised when a configuration is incomplete or inconsistent."""


class ArtifactValidationError(StormPetError):
    """Raised when a model or stage artifact violates its contract."""

