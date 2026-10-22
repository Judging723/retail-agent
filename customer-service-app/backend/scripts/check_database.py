import asyncio

from sqlalchemy import text

from app.core.database import engine


async def main():
    async with engine.connect() as connection:
        version = await connection.scalar(text("SELECT VERSION()"))
        current_user = await connection.scalar(text("SELECT CURRENT_USER()"))
        current_db = await connection.scalar(text("SELECT DATABASE()"))

    print("MySQL version:", version)
    print("Current user:", current_user)
    print("Current database:", current_db)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
