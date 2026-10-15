from app.models import Base, Customer


def test_customer_table_is_registered() -> None:
    assert "customers" in Base.metadata.tables
    assert Base.metadata.tables["customers"] is Customer.__table__


def test_customer_table_columns() -> None:
    columns = Customer.__table__.c

    assert set(columns.keys()) == {
        "id",
        "customer_no",
        "display_name",
        "created_at",
        "updated_at",
    }
    assert columns.id.primary_key
    assert columns.customer_no.nullable is False
    assert columns.customer_no.unique is True
    assert columns.display_name.nullable is True