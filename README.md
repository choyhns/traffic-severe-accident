# 교통사고 중대사고 예측 (Machine Learning)

교통사고 데이터를 이용해 **중대사고 여부**를 예측하는 머신러닝 프로젝트입니다. Streamlit 웹 앱으로 모델 학습·평가·예측을 수행할 수 있습니다.

## 주요 기능

- **데이터**: 교통사고 원본 CSV (23년·24년) 전처리 및 피처 엔지니어링
- **모델**: Logistic Regression, Random Forest, XGBoost 등 이진 분류
- **Streamlit 앱**: 데이터 탐색, 모델 학습/재학습, Feature Importance, 지도 시각화(선택)

## 환경 요구사항

- Python 3.10+
- Windows에서 한글 시각화: **Malgun Gothic** 폰트 (기본 설치)

## 설치 및 실행

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
traffic-severe-accident(ML)/
├── app.py              # Streamlit 메인 앱
├── requirements.txt
├── data/
│   └── raw/            # 사고분석-23년.csv, 사고분석-24년.csv
├── models/
│   └── best_model.pkl  # 저장된 최적 모델 (학습 후 생성)
└── src/
    ├── config.py       # 경로·타겟·컬럼 설정
    ├── io.py           # 데이터 로드
    ├── preprocess.py   # 전처리
    ├── features.py     # 피처/요약 테이블
    ├── models.py       # 모델 빌더 (LR, RF, XGB)
    └── evaluate.py    # 이진 분류 평가
```

## 데이터

- `data/raw/` 에 **사고분석-23년.csv**, **사고분석-24년.csv** 를 두고 사용
- 인코딩 문제 시 `convert_encoding.py` 로 UTF-8 변환 가능

## 라이선스

프로젝트용·학습용으로 자유롭게 사용 가능합니다.
