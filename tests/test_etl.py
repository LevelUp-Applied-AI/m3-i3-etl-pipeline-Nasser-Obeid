"""Tests for the ETL pipeline — Amman Digital Market."""
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from etl_pipeline import transform, validate


def make_data(orders_rows, items_rows):
    customers = pd.DataFrame([
        {"customer_id": 1, "customer_name": "Layla Khalil", "city": "Amman"},
    ])
    products = pd.DataFrame([
        {"product_id": 10, "product_name": "Laptop", "category": "Electronics", "unit_price": 500.00},
    ])
    orders = pd.DataFrame(orders_rows, columns=["order_id", "customer_id", "order_date", "status"])
    items  = pd.DataFrame(items_rows,  columns=["item_id", "order_id", "product_id", "quantity"])
    return {"customers": customers, "products": products, "orders": orders, "order_items": items}


def test_transform_filters_cancelled():
    data = make_data(
        orders_rows=[(1, 1, "2024-01-10", "completed"), (2, 1, "2024-01-15", "cancelled")],
        items_rows= [(1, 1, 10, 1),                     (2, 2, 10, 1)],
    )
    result = transform(data)
    row = result[result["customer_id"] == 1].iloc[0]
    assert row["total_orders"] == 1
    assert row["total_revenue"] == pytest.approx(500.00)


def test_transform_filters_suspicious_quantity():
    data = make_data(
        orders_rows=[(1, 1, "2024-02-01", "completed"), (2, 1, "2024-02-02", "completed")],
        items_rows= [(1, 1, 10, 5),                     (2, 2, 10, 200)],
    )
    result = transform(data)
    row = result[result["customer_id"] == 1].iloc[0]
    assert row["total_revenue"] == pytest.approx(5 * 500.00)


def test_validate_catches_nulls():
    df = pd.DataFrame([{
        "customer_id": None, "customer_name": "Layla Khalil", "city": "Amman",
        "total_orders": 1, "total_revenue": 500.00, "avg_order_value": 500.00,
        "top_category": "Electronics",
    }])
    with pytest.raises(ValueError):
        validate(df)