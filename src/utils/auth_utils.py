"""NewsSnap AI - Authentication and authorization utilities."""

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt

from src.config.settings import settings


def create_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a new JWT token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    # We add an 'iat' (issued at) claim as well
    to_encode.update({"iat": datetime.now(timezone.utc)})

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict[str, Any] | None:
    """Verify a JWT token and return its decoded payload. Returns None if invalid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        return None


def get_google_authorization_url() -> str:
    """Construct the Google OAuth 2.0 authorization URL."""
    base_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{base_url}?{urlencode(params)}"


async def exchange_code_for_profile(code: str) -> dict[str, Any]:
    """Exchange authorization code for tokens and fetch user profile from Google."""
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
    }

    async with httpx.AsyncClient() as client:
        # 1. Exchange code for access token
        token_response = await client.post(token_url, data=token_data)
        token_response.raise_for_status()
        tokens = token_response.json()

        access_token = tokens.get("access_token")
        if not access_token:
            raise ValueError("Failed to obtain access token from Google")

        # 2. Fetch user profile
        userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}

        profile_response = await client.get(userinfo_url, headers=headers)
        profile_response.raise_for_status()

        return profile_response.json()
