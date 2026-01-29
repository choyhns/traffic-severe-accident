from pathlib import Path
import pandas as pd

def _read_csv_safely(path: Path) -> pd.DataFrame:
    # 인코딩 자동 시도 (UTF-8 -> UTF-8-SIG -> CP949 -> EUC-KR)
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 로딩 실패: {path.name} / 마지막 에러: {last_err}")

def load_raw_files(raw_dir: Path, patterns=None) -> pd.DataFrame:
    """
    data/raw 아래의 csv를 읽어 합친다.
    patterns가 None이면 '*.csv' 전부 로딩
    """
    if patterns is None:
        patterns = ["*.csv"]

    dfs = []
    for pat in patterns:
        for p in sorted(raw_dir.glob(pat)):
            df = _read_csv_safely(p)
            df["source_file"] = p.name
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"{raw_dir} 에서 CSV를 찾지 못했습니다.")

    return pd.concat(dfs, ignore_index=True)
