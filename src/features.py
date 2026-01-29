import pandas as pd

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_rate": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        })
        .sort_values("missing_rate", ascending=False)
    )

def severe_rate_by(df: pd.DataFrame, col: str, target="중대사고", min_count=50) -> pd.DataFrame:
    tmp = df.groupby(col)[target].agg(["count", "mean"]).reset_index()
    tmp = tmp[tmp["count"] >= min_count].sort_values("mean", ascending=False)
    tmp = tmp.rename(columns={"mean": "severe_rate"})
    return tmp
