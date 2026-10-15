from pydantic import SecretStr

from app.core.config import Settings
from app.core.database import build_database_url


def test_build_database_url() -> None:
    settings = Settings(
        db_host="database.example",
        db_port=3306,
        db_name="test_db",
        db_user="test_user",
        db_password=SecretStr("special@password"),
        _env_file=None,
    )
    url = build_database_url(settings)

    assert url.drivername == "mysql+asyncmy"
    assert url.host == "database.example"
    assert url.port == 3306
    assert url.database == "test_db"
    assert url.username == "test_user"
    assert url.password == "special@password"
    assert url.query["charset"] == "utf8mb4"

    rendered = url.render_as_string(hide_password=True)

    assert "special@password" not in rendered
    assert "***" in rendered
