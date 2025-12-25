import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CEFR_PATH = DATA_DIR / "oxford_cerf.csv"


def clean_word(w):
    """Chuẩn hóa từ: bỏ dấu câu."""
    return re.sub(r"[^a-zA-Z']", "", str(w)).lower()


def load_cefr_dict(path: Path):
    df = pd.read_csv(path)

    # chuẩn hoá tên cột
    df.columns = [c.lower().strip() for c in df.columns]

    word_col = None
    level_col = None

    for c in df.columns:
        if c in ["word", "headword", "lemma"]:
            word_col = c
        if c in ["cefr", "cefr_level", "level"]:
            level_col = c

    if word_col is None or level_col is None:
        raise ValueError(
            f"CEFR CSV must contain word & level columns, got {df.columns}"
        )

    return dict(
        zip(
            df[word_col].astype(str).str.lower(),
            df[level_col].astype(str).str.upper()
        )
    )


def compute_lexical_score(df: pd.DataFrame):
    """
    Compute lexical CEFR distribution from ASR transcript DataFrame

    Required columns:
        - word
    """

    if df is None or df.empty:
        raise ValueError("Empty transcript DataFrame")

    if "word" not in df.columns:
        raise ValueError("DataFrame must contain 'word' column")

    cefr_dict = load_cefr_dict(CEFR_PATH)

    df = df.copy()

    # Clean word & map CEFR
    df["word_clean"] = df["word"].apply(clean_word)
    df["cefr_level"] = df["word_clean"].apply(
        lambda w: cefr_dict.get(w, "UNK")
    )

    # 👇 chỉ giữ A1–C1
    df = df[df["cefr_level"].isin(["A1", "A2", "B1", "B2", "C1"])]

    if df.empty:
        return {
            "A1": 0.0,
            "A2": 0.0,
            "B1": 0.0,
            "B2": 0.0,
            "C1": 0.0,
        }

    result_df = (
        df.groupby("cefr_level")
          .size()
          .reset_index(name="count")
    )

    result_df["proportion"] = (
        result_df["count"] / result_df["count"].sum() * 100
    )

    def get_level(level):
        row = result_df[result_df["cefr_level"] == level]
        return float(row["proportion"].values[0]) if not row.empty else 0.0

    return {
        "A1": get_level("A1"),
        "A2": get_level("A2"),
        "B1": get_level("B1"),
        "B2": get_level("B2"),
        "C1": get_level("C1"),
    }
