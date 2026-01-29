import re
import numpy as np
import pandas as pd
from .config import TARGET_COL

def _to_int_series(s: pd.Series) -> pd.Series:
    # "1,234" 같은 문자열도 숫자로
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0).astype(int)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    발생년월: '2023년 1월' 형태에서 발생년/발생월 파생
    """
    out = df.copy()
    if "발생년월" in out.columns:
        # 연도
        year = out["발생년월"].astype(str).str.extract(r"(\d{4})")[0]
        out["발생년"] = pd.to_numeric(year, errors="coerce").fillna(-1).astype(int)

        # 월
        month = out["발생년월"].astype(str).str.extract(r"(\d{1,2})\s*월")[0]
        out["발생월"] = pd.to_numeric(month, errors="coerce").fillna(-1).astype(int)
    else:
        out["발생년"] = -1
        out["발생월"] = -1

    # -1(파싱 실패) 제거/보정은 필요하면 여기서
    return out

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    중대사고 = (사망자수 + 중상자수) >= 1
    """
    out = df.copy()

    for c in ["사망자수", "중상자수", "경상자수", "부상신고자수"]:
        if c in out.columns:
            out[c] = _to_int_series(out[c])

    if "사망자수" in out.columns and "중상자수" in out.columns:
        out[TARGET_COL] = ((out["사망자수"] + out["중상자수"]) >= 1).astype(int)
    else:
        raise KeyError("타겟 생성에 필요한 컬럼(사망자수/중상자수)이 없습니다.")

    # 참고용 총피해자수
    if all(c in out.columns for c in ["사망자수", "중상자수", "경상자수", "부상신고자수"]):
        out["총피해자수"] = out["사망자수"] + out["중상자수"] + out["경상자수"] + out["부상신고자수"]

    return out

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    공백 정리 + 문자열 컬럼 정리(앞뒤 공백)
    """
    out = df.copy()

    # 컬럼명 공백 제거(혹시 모를 케이스)
    out.columns = [c.strip() for c in out.columns]

    # object 컬럼 앞뒤 공백 제거
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()

    return out

def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 전처리 파이프라인
    """
    df = basic_clean(df)
    df = add_time_features(df)
    df = add_targets(df)
    return df
