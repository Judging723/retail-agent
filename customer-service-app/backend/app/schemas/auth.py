from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

CustomerNumber = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]
DisplayName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]


class RegisterRequest(BaseModel):
    customer_no: CustomerNumber
    display_name: DisplayName | None = None
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    customer_no: CustomerNumber
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class CurrentCustomerResponse(BaseModel):
    customer_no: str
    display_name: str | None
