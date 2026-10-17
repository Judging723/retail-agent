from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Customer


class CustomerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_customer_no(
        self,
        customer_no: str,
    ) -> Customer | None:
        statement = select(Customer).where(
            Customer.customer_no == customer_no
        )
        return await self._session.scalar(statement)

    async def create(
        self,
        *,
        customer_no: str,
        display_name: str | None,
    ) -> Customer:
        customer = Customer(
            customer_no=customer_no,
            display_name=display_name,
        )
        self._session.add(customer)
        await self._session.flush()
        return customer