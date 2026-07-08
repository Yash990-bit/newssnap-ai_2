"""NewsSnap AI - API module for users."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.middleware import get_current_user
from src.config.database import get_db
from src.config.settings import Category, Language
from src.models.user import User

router = APIRouter(prefix="/api/users", tags=["users"])


class UserProfileResponse(BaseModel):
    id: str
    email: str
    name: str
    is_onboarded: bool
    primary_language: Language | None
    secondary_language: Language | None
    categories: list[str]

    class Config:
        from_attributes = True


class OnboardingRequest(BaseModel):
    primary_language: Language
    secondary_language: Language | None = None
    categories: list[Category] = Field(min_length=3)


@router.get("/me", response_model=UserProfileResponse)
def get_user_me(current_user: User = Depends(get_current_user)):
    """Get the current user's profile and preferences."""
    prefs = current_user.preferences
    return UserProfileResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        is_onboarded=current_user.is_onboarded,
        primary_language=prefs.primary_language if prefs else None,
        secondary_language=prefs.secondary_language if prefs else None,
        categories=prefs.categories if prefs else [],
    )


@router.post("/onboarding", response_model=UserProfileResponse)
def user_onboarding(
    request: OnboardingRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Complete user onboarding by setting languages and categories."""
    if len(request.categories) < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least 3 categories must be selected")

    prefs = current_user.preferences
    if prefs:
        prefs.primary_language = request.primary_language
        prefs.secondary_language = request.secondary_language
        prefs.categories = [cat.value for cat in request.categories]

    current_user.is_onboarded = True
    db.commit()
    db.refresh(current_user)

    return UserProfileResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        is_onboarded=current_user.is_onboarded,
        primary_language=prefs.primary_language if prefs else None,
        secondary_language=prefs.secondary_language if prefs else None,
        categories=prefs.categories if prefs else [],
    )
