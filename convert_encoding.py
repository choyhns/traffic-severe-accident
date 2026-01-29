from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

def read_with_fallback(path: Path) -> tuple[str, str]:
    last_err = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err

def convert_file(path: Path) -> None:
    text, used = read_with_fallback(path)
    # UTF-8로 통일 저장 (엑셀 호환이 필요하면 utf-8-sig 추천)
    path.write_text(text, encoding="utf-8")
    print(f"OK  : {path.name}  ({used} -> utf-8)")

def main():
    if not RAW_DIR.exists():
        print("RAW_DIR not found:", RAW_DIR)
        return

    files = list(RAW_DIR.glob("*.csv"))
    print("RAW_DIR:", RAW_DIR)
    print("CSV files:", len(files))

    for path in files:
        try:
            convert_file(path)
        except Exception as e:
            print(f"FAIL: {path.name}  ({type(e).__name__}: {e})")

    print("Done.")

if __name__ == "__main__":
    main()
