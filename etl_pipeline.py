"""ETL Pipeline — Amman Digital Market Customer Analytics

Extracts data from PostgreSQL, transforms it into customer-level summaries,
validates data quality, and loads results to a database table and CSV file.
"""
from sqlalchemy import create_engine
import pandas as pd
import os


def extract(engine):
    """Extract all source tables from PostgreSQL into DataFrames.

    Args:
        engine: SQLAlchemy engine connected to the amman_market database

    Returns:
        dict: {"customers": df, "products": df, "orders": df, "order_items": df}
    """
    tables = ["customers", "products", "orders", "order_items"]
    data = {}

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        data[table] = df
        print(f"  [extract] {table}: {len(df):,} rows loaded")

    return data


def transform(data_dict):
    """Transform raw data into customer-level analytics summary.

    Steps:
    1. Join orders with order_items and products
    2. Compute line_total (quantity * unit_price)
    3. Filter out cancelled orders (status = 'cancelled')
    4. Filter out suspicious quantities (quantity > 100)
    5. Aggregate to customer level: total_orders, total_revenue,
       avg_order_value, top_category

    Args:
        data_dict: dict of DataFrames from extract()

    Returns:
        DataFrame: customer-level summary with columns:
            customer_id, customer_name, city, total_orders,
            total_revenue, avg_order_value, top_category
    """
    customers   = data_dict["customers"]
    products    = data_dict["products"]
    orders      = data_dict["orders"]
    order_items = data_dict["order_items"]

    # Step 1 — join order_items → orders → products
    df = (
        order_items
        .merge(orders,   on="order_id",   how="inner")
        .merge(products, on="product_id", how="inner")
    )

    # Step 2 — line total per item row
    df["line_total"] = df["quantity"] * df["unit_price"]

    # Step 3 — drop cancelled orders
    before = len(df)
    df = df[df["status"] != "cancelled"]
    print(f"  [transform] cancelled-order rows dropped: {before - len(df):,}")

    # Step 4 — drop suspicious quantities
    before = len(df)
    df = df[df["quantity"] <= 100]
    print(f"  [transform] suspicious-quantity rows dropped: {before - len(df):,}")

    # Step 5a — top category per customer (highest total revenue by category)
    category_revenue = (
        df.groupby(["customer_id", "category"], as_index=False)["line_total"]
        .sum()
    )
    top_category = (
        category_revenue
        .sort_values("line_total", ascending=False)
        .drop_duplicates(subset="customer_id", keep="first")
        [["customer_id", "category"]]
        .rename(columns={"category": "top_category"})
    )

    # Step 5b — aggregate to customer level
    summary = (
        df.groupby("customer_id", as_index=False)
        .agg(
            total_orders=("order_id",   pd.Series.nunique),
            total_revenue=("line_total", "sum"),
        )
    )
    summary["avg_order_value"] = summary["total_revenue"] / summary["total_orders"]

    # Step 5c — attach top_category and customer details
    summary = (
        summary
        .merge(top_category, on="customer_id", how="left")
        .merge(customers[["customer_id", "customer_name", "city"]],
               on="customer_id", how="left")
    )

    # Reorder columns to match spec
    summary = summary[[
        "customer_id", "customer_name", "city",
        "total_orders", "total_revenue", "avg_order_value", "top_category",
    ]]

    print(f"  [transform] customer summary rows produced: {len(summary):,}")
    return summary


def validate(df):
    """Run data quality checks on the transformed DataFrame.

    Checks:
    - No nulls in customer_id or customer_name
    - total_revenue > 0 for all customers
    - No duplicate customer_ids
    - total_orders > 0 for all customers

    Args:
        df: transformed customer summary DataFrame

    Returns:
        dict: {check_name: bool} for each check

    Raises:
        ValueError: if any critical check fails
    """
    checks = {}

    # Check 1 — no nulls in customer_id
    checks["no_null_customer_id"] = bool(df["customer_id"].notna().all())

    # Check 2 — no nulls in customer_name
    checks["no_null_customer_name"] = bool(df["customer_name"].notna().all())

    # Check 3 — total_revenue > 0
    checks["positive_total_revenue"] = bool((df["total_revenue"] > 0).all())

    # Check 4 — no duplicate customer_ids
    checks["no_duplicate_customer_id"] = bool(df["customer_id"].nunique() == len(df))

    # Check 5 — total_orders > 0
    checks["positive_total_orders"] = bool((df["total_orders"] > 0).all())

    # Report
    print()
    print("  [validate] Data quality report:")
    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {status}  {check}")
        if not passed:
            all_passed = False

    if not all_passed:
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(
            f"[validate] Critical checks failed: {', '.join(failed)}"
        )

    print("  [validate] All checks passed.\n")
    return checks


def load(df, engine, csv_path):
    """Load customer summary to PostgreSQL table and CSV file.

    Args:
        df: validated customer summary DataFrame
        engine: SQLAlchemy engine
        csv_path: path for CSV output
    """
    # Write to PostgreSQL
    df.to_sql("customer_analytics", engine, if_exists="replace", index=False)

    # Ensure output directory exists, then write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"  [load] {len(df):,} rows written to PostgreSQL table 'customer_analytics'")
    print(f"  [load] {len(df):,} rows written to {csv_path}")


def main():
    """Orchestrate the ETL pipeline: extract -> transform -> validate -> load."""
    db_url   = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/amman_market")
    csv_path = "output/customer_analytics.csv"

    print("=" * 55)
    print("  Amman Digital Market — Customer Analytics ETL")
    print("=" * 55)

    # 1. Connect
    print("\n[1/4] Connecting to database …")
    engine = create_engine(db_url)

    # 2. Extract
    print("\n[2/4] Extracting source tables …")
    data = extract(engine)
    total_source_rows = sum(len(v) for v in data.values())
    print(f"  Total source rows: {total_source_rows:,}")

    # 3. Transform
    print("\n[3/4] Transforming data …")
    summary = transform(data)

    # 4. Validate
    print("\n[4/4] Validating …")
    validate(summary)

    # 5. Load
    print("[5/5] Loading results …")
    load(summary, engine, csv_path)

    print("\nPipeline complete.")
    print(f"  Customer records written: {len(summary):,}")
    print("=" * 55)


if __name__ == "__main__":
    main()