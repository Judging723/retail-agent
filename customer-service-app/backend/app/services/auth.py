from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.models import Customer


class CustomerAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register(
        self,
        *,
        customer_no: str,
        display_name: str | None,
        password: str,
    ) -> Customer:
        try:
            async with self._session.begin():
                existing = await self._session.scalar(
                    select(Customer).where(Customer.customer_no == customer_no)
                )
                if existing is not None:
                    raise CustomerAlreadyExistsError

                customer = Customer(
                    customer_no=customer_no,
                    display_name=display_name,
                    password_hash=hash_password(password),
                )
                self._session.add(customer)
                await self._session.flush()
                await self._session.refresh(customer)
        except IntegrityError as exc:
            raise CustomerAlreadyExistsError from exc

        return customer

    async def authenticate(self, *, customer_no: str, password: str) -> Customer:
        async with self._session.begin():
            customer = await self._session.scalar(
                select(Customer).where(Customer.customer_no == customer_no)
            )

        if (
            customer is None
            or customer.password_hash is None
            or not verify_password(password, customer.password_hash)
        ):
            raise InvalidCredentialsError

        return customer
