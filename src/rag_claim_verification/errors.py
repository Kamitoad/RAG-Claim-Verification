"""Application-specific exceptions with user-facing messages."""


class RagClaimVerificationError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(RagClaimVerificationError):
    """Raised when a configuration is invalid or inconsistent."""


class ManifestError(RagClaimVerificationError):
    """Raised when a document manifest or its files are invalid."""


class ExternalDependencyError(RagClaimVerificationError):
    """Raised when an optional external integration is unavailable."""


class ProviderError(RagClaimVerificationError):
    """Raised when an LLM or embedding provider request fails."""
