"""Typed pandas frames matching the output-table schemas."""

from typing import Any

import pandas as pd


def typed_frame(rows: list[dict[str, Any]], columns: dict[str, str]) -> pd.DataFrame:
    """Build a frame whose dtypes match a column spec, keeping nulls as nulls.

    Nullable pandas dtypes are used so integer and boolean columns can hold
    missing values; Spark's Arrow conversion maps them to nullable columns.

    Parameters
    ----------
    rows : list of dict
        One dict per row; missing keys become null.
    columns : dict of str to str
        Column name to type name (``string``, ``int``, ``double``,
        ``boolean`` or ``timestamp``), in output order.

    Returns
    -------
    pd.DataFrame
        Frame with exactly the requested columns and dtypes.
    """
    data: dict[str, Any] = {}
    for name, dtype in columns.items():
        values = [row.get(name) for row in rows]
        if dtype == "int":
            data[name] = pd.array([None if v is None or pd.isna(v) else int(v) for v in values], dtype="Int64")
        elif dtype == "boolean":
            data[name] = pd.array([None if v is None or pd.isna(v) else bool(v) for v in values], dtype="boolean")
        elif dtype == "double":
            data[name] = pd.array([float("nan") if v is None else float(v) for v in values], dtype="float64")
        elif dtype == "timestamp":
            data[name] = pd.to_datetime(pd.Series(values, dtype="object"))
        else:
            data[name] = pd.array([None if v is None else str(v) for v in values], dtype="object")
    return pd.DataFrame(data, columns=list(columns))
