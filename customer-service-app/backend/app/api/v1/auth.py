from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import AuthServiceDependency, CurrentCustomerDependency
from app.core.config import get_settings
from app.core.security import create_access_token
from app.schemas.auth import (
    CurrentCustomerResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.services import CustomerAlreadyExistsError, InvalidCredentialsError

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(customer_id: int, customer_no: str) -> TokenResponse:
    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(
            customer_id=customer_id,
            customer_no=customer_no,
            settings=settings,
        ),
        expires_in=settings.jwt_access_token_minutes * 60,
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(payload: RegisterRequest, service: AuthServiceDependency) -> TokenResponse:
    try:
        customer = await service.register(
            customer_no=payload.customer_no,
            display_name=payload.display_name,
            password=payload.password,
        )
    except CustomerAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Customer number is already registered",
        ) from exc

    return _token_response(customer.id, customer.customer_no)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, service: AuthServiceDependency) -> TokenResponse:
    try:
        customer = await service.authenticate(
            customer_no=payload.customer_no,
            password=payload.password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer number or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return _token_response(customer.id, customer.customer_no)


@router.get("/me", response_model=CurrentCustomerResponse)
async def get_me(customer: CurrentCustomerDependency) -> CurrentCustomerResponse:
    return CurrentCustomerResponse(
        customer_no=customer.customer_no,
        display_name=customer.display_name,
    )
