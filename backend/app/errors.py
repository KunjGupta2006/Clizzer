class ClizzerError(Exception):
    """Base exception for all Clizzer errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SourceUnavailable(ClizzerError):
    """Raised when source video cannot be downloaded, accessed, or is corrupted."""
    pass


class NoSpeech(ClizzerError):
    """Raised when no speech is detected in the input audio."""
    pass


class ChatUnavailable(ClizzerError):
    """Raised when live chat cannot be fetched or is absent."""
    pass


class RenderFailed(ClizzerError):
    """Raised when video re-encoding or clipping ffmpeg pipeline fails."""
    pass


class LLMUnavailable(ClizzerError):
    """Raised when LLM API exhausts retries or is completely unreachable."""
    pass
