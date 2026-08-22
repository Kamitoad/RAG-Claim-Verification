"""Shared model configuration."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and trims input strings."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )
