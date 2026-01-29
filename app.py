# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import (
    RAW_DIR, MODELS_DIR, TARGET_COL,
    DEFAULT_FINAL_COLS, OPTIONAL_COLS, LEAKAGE_OR_DROP_COLS
)
from src.io import load_raw_files
from src.preprocess import preprocess_all
from src.features import summary_table, severe_rate_by
from src.models import build_logistic, build_random_forest
from src.evaluate import evaluate_binary

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# Streamlit 설정
# -----------------------------
st.set_page_config(page_title="교통사고 중대사고 예측", layout="wide")
st.title("🚦 교통사고 중대사고 예측 (Streamlit)")

MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------
# 캐시: 데이터 로딩/전처리
# -----------------------------
@st.cache_data
def load_and_preprocess():
    df = load_raw_files(RAW_DIR, patterns=["*.csv"])
    df = preprocess_all(df)
    return df

# -----------------------------
# 캐시: 모델 로드 (앱 실행 중 1회)
# -----------------------------
@st.cache_resource
def load_saved_model(path: str):
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    return None

# -----------------------------
# 유틸: 파이프라인 생성
# -----------------------------
def build_pipeline(model, cat_cols):
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    return pipe

def try_build_xgboost(scale_pos_weight: float):
    try:
        from src.models import build_xgboost
        return build_xgboost(scale_pos_weight)
    except Exception:
        return None

# -----------------------------
# 유틸: feature importance 추출
# -----------------------------
def get_encoded_feature_names(pipe):
    # pipe: Pipeline(preprocess=ColumnTransformer, model=...)
    ct = pipe.named_steps["preprocess"]
    names = ct.get_feature_names_out()
    # 보기 좋게 prefix 정리 (cat__ 제거)
    names = [n.replace("cat__", "") for n in names]
    return names


def compute_feature_importance(pipe: Pipeline, cat_cols: list[str]) -> pd.DataFrame:
    """
    모델별 중요도:
    - RF/XGB: feature_importances_
    - Logistic: |coef_| 합(다중 클래스 방어)
    """
    model = pipe.named_steps["model"]
    feat_names = get_encoded_feature_names(pipe)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            imp = np.abs(coef).sum(axis=0)
        else:
            imp = np.abs(coef)
    else:
        imp = np.zeros(len(feat_names), dtype=float)

    imp_df = pd.DataFrame({"feature": feat_names, "importance": imp})
    imp_df["base_col"] = imp_df["feature"].astype(str).str.split("_", n=1).str[0]
    col_imp = (
        imp_df.groupby("base_col")["importance"]
        .sum()
        .reset_index()
        .rename(columns={"importance": "model_importance"})
        .sort_values("model_importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True), col_imp

def ensure_model_loaded_to_state():
    """앱 시작 시 저장 모델이 있으면 session_state에 올려두기"""
    if "model" not in st.session_state or st.session_state["model"] is None:
        saved = load_saved_model(str(MODEL_PATH))
        if saved is not None:
            st.session_state["model"] = saved
            st.session_state["model_name"] = "best_model.pkl(loaded)"
            st.session_state["model_path"] = str(MODEL_PATH)

# -----------------------------
# 데이터 준비
# -----------------------------
with st.spinner("데이터 로딩/전처리 중..."):
    acc = load_and_preprocess()
ensure_model_loaded_to_state()

# -----------------------------
# 탭 6개 구성
# -----------------------------
t1, t2, t3, t4, t5, t6 = st.tabs([
    "1) 데이터 개요",
    "2) 범주별 중대사고율",
    "3) 모델 학습/성능 비교",
    "4) 예측",
    "5) Feature Importance",
    "6) 저장/불러오기",
])

# =========================================================
# 1) 데이터 개요
# =========================================================
with t1:
    st.subheader("데이터 개요")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**데이터 구조/결측치 요약**")
        st.dataframe(summary_table(acc), use_container_width=True)
    with c2:
        st.write("**기본 통계**")
        st.metric("총 행 수", f"{len(acc):,}")
        st.metric("중대사고 비율(Mean)", f"{float(acc[TARGET_COL].mean()*100):.1f}%")
        st.write("- 타겟 정의: (사망자수 + 중상자수) ≥ 1 → 중대사고=1")
        

    st.divider()
    st.write("**누수/식별자/사후정보로 제외 후보 컬럼(참고)**")
    st.write([c for c in LEAKAGE_OR_DROP_COLS if c in acc.columns])

# =========================================================
# 2) 범주별 중대사고율
# =========================================================
with t2:
    st.subheader("범주별 중대사고율 비교")

    # 범주형 후보
    cat_candidates = [c for c in acc.columns if acc[c].dtype == "object"]
    default_idx = cat_candidates.index("사고유형") if "사고유형" in cat_candidates else 0
    col = st.selectbox("변수 선택", cat_candidates, index=default_idx)

    min_count = st.slider("최소 표본수(min_count) 이하 범주는 제외", 10, 500, 50, 10)

    rate_df = severe_rate_by(acc, col, target=TARGET_COL, min_count=min_count)
    st.dataframe(rate_df.head(30), use_container_width=True)

    if len(rate_df) > 0:
        topn = st.slider("그래프 표시 범주 수(상위)", 5, 30, 12, 1)
        plot_df = rate_df.head(topn).sort_values("severe_rate")
        fig = plt.figure(figsize=(8, 5))
        plt.barh(plot_df[col].astype(str), plot_df["severe_rate"])
        plt.title(f"{col}별 중대사고율 (상위 {topn})")
        plt.xlabel("중대사고율")
        plt.tight_layout()
        st.pyplot(fig)

# =========================================================
# 3) 모델 학습/성능 비교
# =========================================================
with t3:
    st.subheader("모델 학습 및 성능 비교 (Logistic / RandomForest / XGBoost)")

    st.write("**학습 변수 선택**")
    use_optional = st.checkbox("보조 변수도 포함(주야/노면/기상 등)", value=False)
    feature_cols = DEFAULT_FINAL_COLS + OPTIONAL_COLS if use_optional else DEFAULT_FINAL_COLS
    feature_cols = [c for c in feature_cols if c in acc.columns]

    st.caption(f"선택된 변수: {feature_cols}")

    test_size = st.slider("test_size", 0.1, 0.4, 0.2, 0.05)
    metric_pick = st.selectbox("Best 모델 기준", ["f1", "auc"], index=0)

    with st.expander("왜 일부 컬럼은 학습에서 제외하나요?"):
        st.write("- 사고 발생 시점에 알기 어려운 결과/사후 정보 또는 식별자(구분번호) 등은 모델이 답을 '미리' 알아버리는 누수 위험이 있습니다.")
        st.write([c for c in LEAKAGE_OR_DROP_COLS if c in acc.columns])

    run_train = st.button("✅ 3개 모델 학습/평가 실행", type="primary")

    if run_train:
        X = acc[feature_cols].copy()
        y = acc[TARGET_COL].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        results = []

        # Logistic
        lr = build_logistic()
        lr_pipe = build_pipeline(lr, feature_cols)
        with st.spinner("Logistic Regression 학습 중..."):
            lr_pipe.fit(X_train, y_train)
        lr_metrics = evaluate_binary(lr_pipe, X_test, y_test)
        results.append(("LogisticRegression", lr_pipe, lr_metrics))

        # RandomForest
        rf = build_random_forest()
        rf_pipe = build_pipeline(rf, feature_cols)
        with st.spinner("RandomForest 학습 중..."):
            rf_pipe.fit(X_train, y_train)
        rf_metrics = evaluate_binary(rf_pipe, X_test, y_test)
        results.append(("RandomForest", rf_pipe, rf_metrics))

        # XGBoost (있으면)
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        xgb = try_build_xgboost(neg / pos)
        if xgb is None:
            st.warning("XGBoost 미설치로 스킵했습니다. 필요 시: pip install xgboost")
        else:
            xgb_pipe = build_pipeline(xgb, feature_cols)
            with st.spinner("XGBoost 학습 중..."):
                xgb_pipe.fit(X_train, y_train)
            xgb_metrics = evaluate_binary(xgb_pipe, X_test, y_test)
            results.append(("XGBoost", xgb_pipe, xgb_metrics))

        # 결과 표
        rows = []
        for name, _, m in results:
            rows.append({
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auc": m["auc"],
            })
        res_df = pd.DataFrame(rows).sort_values(metric_pick, ascending=False).reset_index(drop=True)
        st.dataframe(res_df, use_container_width=True)


        st.subheader("모델별 Recall / F1 비교")

        plot_df = res_df.set_index("model")[["recall", "f1"]]

        fig = plt.figure(figsize=(7, 4))
        plot_df.plot(kind="bar", ax=plt.gca())
        plt.ylim(0, 1)
        plt.ylabel("score")
        plt.title("Model Performance Comparison (Recall & F1)")
        plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Recall은 중대사고를 놓치지 않는 능력, "
            "F1은 Recall과 Precision의 균형을 의미합니다."
        )

        st.subheader("모델별 AUC 비교")

        auc_df = res_df.set_index("model")[["auc"]]

        fig = plt.figure(figsize=(6, 4))
        auc_df.plot(kind="bar", ax=plt.gca(), legend=False)
        plt.ylim(0, 1)
        plt.ylabel("AUC")
        plt.title("Model Performance Comparison (AUC)")
        plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "AUC는 임계값과 무관하게 모델의 전체적인 분류 성능을 평가하는 지표입니다."
        )
        
        # best 모델 저장(세션에)
        best_name = res_df.loc[0, "model"]
        best_pipe, best_metrics = None, None
        for name, pipe, m in results:
            if name == best_name:
                best_pipe, best_metrics = pipe, m
                break

        st.session_state["model"] = best_pipe
        st.session_state["model_name"] = best_name
        st.session_state["model_metrics"] = best_metrics
        st.session_state["feature_cols"] = feature_cols
        st.session_state["compare_table"] = res_df

        st.success(f"Best 모델: **{best_name}** (기준: {metric_pick})")

        # Confusion Matrix
        cm = best_metrics["confusion_matrix"]
        fig = plt.figure(figsize=(4, 3))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix ({best_name})")
        plt.xlabel("Pred")
        plt.ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        st.pyplot(fig)

    # 이미 학습된 결과가 있으면 표시
    if "compare_table" in st.session_state:
        st.divider()
        st.write("**최근 학습 결과(세션)**")
        st.dataframe(st.session_state["compare_table"], use_container_width=True)

# =========================================================
# 4) 예측
# =========================================================
with t4:
    st.subheader("단건 예측 / 배치 예측")

    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("먼저 [3) 모델 학습/성능 비교] 또는 [6) 저장/불러오기]에서 모델을 준비하세요.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state.get("feature_cols", DEFAULT_FINAL_COLS)
        feature_cols = [c for c in feature_cols if c in acc.columns]

        st.write("### ① 단건 예측")
        input_row = {}
        cols = st.columns(3)
        for i, c in enumerate(feature_cols):
            opts = sorted(acc[c].dropna().astype(str).unique().tolist())
            if "미상" not in opts:
                opts = ["미상"] + opts
            input_row[c] = cols[i % 3].selectbox(c, opts, index=0, key=f"pred_{c}")

        X_one = pd.DataFrame([input_row])

        if st.button("🔍 예측 실행", type="primary"):
            pred = int(model.predict(X_one)[0])
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = float(model.predict_proba(X_one)[:, 1][0])
                except Exception:
                    proba = None

            st.write(f"예측 결과(중대사고): **{pred}**")
            if proba is not None:
                st.write(f"중대사고 확률: **{proba:.3f}**")

        st.divider()
        st.write("### ② CSV 배치 예측")
        up = st.file_uploader("예측용 CSV 업로드", type=["csv"])
        if up is not None:
            df_in = pd.read_csv(up)
            need = [c for c in feature_cols if c in df_in.columns]
            if len(need) != len(feature_cols):
                st.error("업로드 CSV에 필요한 feature 컬럼이 부족합니다.")
                st.write("필요 컬럼:", feature_cols)
            else:
                Xb = df_in[feature_cols].copy()
                for c in feature_cols:
                    Xb[c] = Xb[c].astype(object).fillna("미상")

                pred_b = model.predict(Xb)
                out = df_in.copy()
                out["pred_중대사고"] = pred_b

                if hasattr(model, "predict_proba"):
                    try:
                        out["proba_중대사고"] = model.predict_proba(Xb)[:, 1]
                    except Exception:
                        pass

                st.dataframe(out.head(30), use_container_width=True)
                st.download_button(
                    "결과 CSV 다운로드",
                    data=out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# =========================================================
# 5) Feature Importance
# =========================================================
with t5:
    st.subheader("Feature Importance (One-Hot 기준)")

    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("먼저 [3) 모델 학습/성능 비교]에서 모델을 학습하거나 [6) 저장/불러오기]로 로드하세요.")
    else:
        model_pipe = st.session_state["model"]
        feature_cols = st.session_state.get("feature_cols", DEFAULT_FINAL_COLS)
        feature_cols = [c for c in feature_cols if c in acc.columns]

        imp_df, col_imp = compute_feature_importance(model_pipe, feature_cols)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**원본 컬럼 단위 중요도 합(추천)**")
            st.dataframe(col_imp, use_container_width=True)

            topn = st.slider("컬럼 중요도 그래프 Top-N", 3, 20, 10, 1)
            plot_df = col_imp.head(topn).sort_values("model_importance")
            fig = plt.figure(figsize=(7, 4))
            plt.barh(plot_df["base_col"], plot_df["model_importance"])
            plt.title(f"컬럼 중요도 Top {topn}")
            plt.xlabel("importance(합)")
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            st.write("**One-Hot 피처 단위 중요도(상위)**")
            st.dataframe(imp_df.head(30), use_container_width=True)

            topn2 = st.slider("One-Hot 중요도 그래프 Top-N", 5, 30, 15, 1)
            plot2 = imp_df.head(topn2).sort_values("importance")
            fig2 = plt.figure(figsize=(8, 5))
            plt.barh(plot2["feature"], plot2["importance"])
            plt.title(f"One-Hot Feature Importance Top {topn2}")
            plt.xlabel("importance")
            plt.tight_layout()
            st.pyplot(fig2)

        st.caption("※ LogisticRegression의 경우 |coef| 기반 중요도(절대값), RandomForest/XGBoost는 feature_importances_ 기반입니다.")

# =========================================================
# 6) 저장/불러오기
# =========================================================
with t6:
    st.subheader("모델 저장 및 불러오기")

    st.write("현재 모델 상태:")
    if "model" in st.session_state and st.session_state["model"] is not None:
        st.success(f"- 로드됨: {st.session_state.get('model_name', '(unknown)')}")
        if "model_metrics" in st.session_state and st.session_state["model_metrics"] is not None:
            m = st.session_state["model_metrics"]
            st.write(f"- f1={m['f1']:.3f}, auc={m['auc']:.3f} (NaN일 수 있음), recall={m['recall']:.3f}")
        st.write("- feature_cols:", st.session_state.get("feature_cols", DEFAULT_FINAL_COLS))
    else:
        st.warning("- 아직 모델이 없습니다. [3) 모델 학습/성능 비교]에서 학습하거나 저장 모델을 불러오세요.")

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("💾 best_model.pkl 저장", type="primary"):
            if "model" not in st.session_state or st.session_state["model"] is None:
                st.warning("저장할 모델이 없습니다. 먼저 학습/로드하세요.")
            else:
                joblib.dump(st.session_state["model"], MODEL_PATH)
                st.success(f"저장 완료: {MODEL_PATH}")

    with c2:
        if st.button("📦 저장 모델 불러오기"):
            if MODEL_PATH.exists():
                m = joblib.load(MODEL_PATH)
                st.session_state["model"] = m
                st.session_state["model_name"] = "best_model.pkl(loaded)"
                st.session_state["model_path"] = str(MODEL_PATH)
                st.success("불러오기 완료!")
            else:
                st.warning("저장된 모델이 없습니다. 먼저 저장하세요.")

    with c3:
        st.write("모델 파일 위치:", str(MODEL_PATH))
        st.write("models 폴더:", str(MODELS_DIR))
