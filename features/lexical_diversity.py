import numpy as np
import re
import pandas as pd

# -------------------------------
# TOKENIZATION (simple)
# -------------------------------
def tokenize(words):
    """
    Input: list of word strings
    Output: list of cleaned tokens (lowercase)
    """
    tokens = []
    for w in words:
        w = str(w).lower()
        w = re.sub(r"[^a-z']", "", w)  # keep alpha + apostrophe
        if w != "":
            tokens.append(w)
    return tokens


# -------------------------------
# TTR (Type-Token Ratio)
# -------------------------------
def compute_ttr(tokens):
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


# -------------------------------
# MSTTR (Mean Segmental TTR)
# -------------------------------
def compute_msttr(tokens, segment_size=50):
    if len(tokens) < segment_size:
        return compute_ttr(tokens)

    ttrs = []
    for i in range(0, len(tokens) - segment_size + 1, segment_size):
        segment = tokens[i:i + segment_size]
        ttrs.append(compute_ttr(segment))

    return float(np.mean(ttrs)) if ttrs else 0.0


# ====================================================
# MAIN WRAPPER (DataFrame version)
# ====================================================
def compute_lexical_diversity_metrics(df: pd.DataFrame):
    """
    Compute lexical diversity metrics from ASR transcript DataFrame

    Required columns:
        - word
    """

    if df is None or df.empty:
        raise ValueError("Empty transcript DataFrame")

    if "word" not in df.columns:
        raise ValueError("DataFrame must contain 'word' column")

    # Extract raw words
    raw_words = df["word"].astype(str).tolist()

    tokens = tokenize(raw_words)

    return {
        "unique_types": len(set(tokens)),
        "total_tokens": len(tokens),
        "TTR": compute_ttr(tokens),
        "MSTTR": compute_msttr(tokens, segment_size=50),
    }
