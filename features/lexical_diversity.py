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
        w = w.lower()
        w = re.sub(r"[^a-z']", "", w)  # keep alpha + apostrophe
        if w != "":
            tokens.append(w)
    return tokens

# -------------------------------
# TTR (Type-Token Ratio)
# -------------------------------
def compute_ttr(tokens):
    if len(tokens) == 0:
        return 0
    types = len(set(tokens))
    tokens_n = len(tokens)
    return types / tokens_n


# -------------------------------
# MSTTR (Mean Segmental TTR)
# -------------------------------
def compute_msttr(tokens, segment_size=50):
    if len(tokens) < segment_size:
        return compute_ttr(tokens)

    ttrs = []
    for i in range(0, len(tokens) - segment_size + 1, segment_size):
        segment = tokens[i:i+segment_size]
        ttrs.append(compute_ttr(segment))

    return float(np.mean(ttrs))

# ====================================================
# MAIN WRAPPER (plug into Whisper result)
# ====================================================
def compute_lexical_diversity_metrics(file):
    """
    result = Whisper output from model.transcribe(..., word_timestamps=True)
    """
    df = pd.read_csv(file)
    raw_words = list(df['word'])

    tokens = tokenize(raw_words)

    return {
        "file": file,
        "unique_types": len(set(tokens)),
        "total_tokens": len(tokens),
        "TTR": compute_ttr(tokens),
        "MSTTR": compute_msttr(tokens, segment_size=50),
    }
    