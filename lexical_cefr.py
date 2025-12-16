import pandas as pd
import re

def clean_word(w):
    """Chuẩn hóa từ: bỏ dấu câu."""
    return re.sub(r"[^a-zA-Z']", "", w).lower()

def load_cefr_dict(path="oxford_cerf.csv"):
    """Tạo dictionary word → CEFR"""
    df = pd.read_csv(path)
    df['word_clean'] = df['word'].str.lower().str.strip()
    return dict(zip(df['word_clean'], df['level']))

def cefr_to_score(level):
    mapping = {
        "A1": 1,
        "A2": 2,
        "B1": 3,
        "B2": 4,
        "C1": 5,
        "C2": 6,
    }
    return mapping.get(level, 0)

def get_prop(i, result_df):
    try:
        return result_df.iloc[i]["proportion"]
    except (IndexError, KeyError):
        return 0

def compute_lexical_score(file):
    words = []
    levels = []
    scores = []
    asr_df = pd.read_csv(file)
    cefr_dict = load_cefr_dict("/content/oxford_cerf.csv")

    for w in asr_df["word"]:
        w_clean = clean_word(w)
        level = cefr_dict.get(w_clean, None)
        score = cefr_to_score(level)

        words.append(w_clean)
        levels.append(level)
        scores.append(score)

    result_df = asr_df.copy()
    result_df["word_clean"] = words
    result_df["cefr_level"] = levels
    result_df["lexical_score"] = scores

    # Tính tổng và trung bình
    result_df = result_df.groupby(['cefr_level']).agg({'lexical_score': 'count'}).reset_index()
    result_df['proportion'] = result_df['lexical_score']/result_df['lexical_score'].sum()*100

    return {
        "file": file,
        "A1": get_prop(0, result_df),
        "A2": get_prop(1, result_df),
        "B1": get_prop(2, result_df),
        "B2": get_prop(3, result_df),
        "C1": get_prop(4, result_df),
    }
