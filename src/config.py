from pathlib import Path

# 프로젝트 루트 기준 경로
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# 기본 타겟/컬럼
TARGET_COL = "중대사고"

# 기본 후보(너가 쓰던 핵심 변수)
DEFAULT_FINAL_COLS = [
    "사고유형",
    "법규위반",
    "가해운전자 차종",
    "도로형태",
    "가해운전자 연령대",
    "가해운전자 성별",
]

# (원하면 추가로 쓰는 보조 변수)
OPTIONAL_COLS = [
    "주야",
    "노면상태",
    "기상상태",
    

]

# 누수/식별자/사후정보 후보
LEAKAGE_OR_DROP_COLS = [
    "구분번호",
    "사고내용",           # 결과를 직접 설명(사후 정보 성격)
    "가해운전자 상해정도", # 사후 결과
    "피해운전자 상해정도", # 사후 결과
    # 아래 피해자수들은 타겟 정의에 사용되는 값 → 학습에는 보통 제외(누수 위험)
    "사망자수",
    "중상자수",
    "경상자수",
    "부상신고자수",
    "총피해자수",
    "피해운전자 차종",
    "피해운전자 연령대",
    "피해운전자 성별",
]
