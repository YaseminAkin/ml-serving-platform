"""Authentication and authorization middleware for the gateway."""

import os

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Validate the X-API-Key header against the configured API key."""
    valid_key = os.environ.get("API_KEY", "dev")

    if x_api_key != valid_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
