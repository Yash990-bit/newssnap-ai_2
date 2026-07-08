"""NewsSnap AI - API module for authentication."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.config.settings import settings
from src.models.user import User, UserPreference
from src.utils.auth_utils import (
    create_token,
    exchange_code_for_profile,
    get_google_authorization_url,
    verify_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.get("/google")
def login_google():
    """Redirects to the Google OAuth 2.0 consent screen."""
    url = get_google_authorization_url()
    return RedirectResponse(url)


@router.get("/google/callback", response_model=TokenResponse)
async def auth_google_callback(code: str, db: Session = Depends(get_db)):
    """Handles the callback from Google OAuth 2.0."""
    try:
        profile = await exchange_code_for_profile(code)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Authentication failed: {str(e)}")

    email = profile.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not provided by Google")

    name = profile.get("name") or profile.get("given_name") or email.split("@")[0]

    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Create new user
        user = User(
            email=email,
            name=name,
            is_active=True,
            is_onboarded=False,
        )
        db.add(user)
        db.flush()  # Flush to get user.id for the preferences

        # Create default preferences
        pref = UserPreference(user_id=user.id)
        db.add(pref)
        db.commit()

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    access_token = create_token({"sub": str(user.id)})
    refresh_token = create_token(
        {"sub": str(user.id), "type": "refresh"}, expires_delta=timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    """Issues a new access token using a valid refresh token."""
    payload = verify_token(request.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()

    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive user")

    access_token = create_token({"sub": str(user.id)})
    refresh_token = create_token(
        {"sub": str(user.id), "type": "refresh"}, expires_delta=timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)
