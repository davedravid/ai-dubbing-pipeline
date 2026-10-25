"""Errors with messages meant to be read by a person, not a programmer."""


class PipelineError(Exception):
    """A problem the user can act on (bad input, missing tool, existing output)."""


class ConfigError(PipelineError):
    """config.yaml is missing, malformed, or has an invalid value."""


class MediaError(PipelineError):
    """FFmpeg is missing, or an input file cannot be read."""


class OutputExistsError(PipelineError):
    """A result file already exists and overwriting was not requested."""
