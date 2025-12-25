import pandas as pd
import numpy as np

def compute_fluency_metrics(df: pd.DataFrame, pause_threshold: float = 0.25):
    """
    Compute:
        - Speech Rate Per Second
        - Ratio pauses To Duration

    Required columns:
        - word
        - start
        - end
    """

    if df is None or df.empty:
        raise ValueError("Empty transcript DataFrame")

    # Ensure correct type
    df = df.copy()
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)

    # Sort by time
    df = df.sort_values("start").reset_index(drop=True)

    # -------------------------
    # Extract words with timestamps
    # -------------------------
    words = []
    for index in range(len(df)):
        words.append({
            "word": df.iloc[index]["word"],
            "start": df.iloc[index]["start"],
            "end": df.iloc[index]["end"]
        })

    if not words:
        raise ValueError("No words found in transcript")

    total_duration = words[-1]["end"]

    if total_duration <= 0:
        raise ValueError("Invalid total duration")

    # -------------------------
    # Speech Rate
    # -------------------------
    total_words = len(words)
    speech_rate_wps = total_words / total_duration

    # -------------------------
    # Pause Detection
    # -------------------------
    pauses = []

    for i in range(1, total_words):
        prev_end = words[i - 1]["end"]
        curr_start = words[i]["start"]
        gap = curr_start - prev_end

        if gap >= pause_threshold:
            pauses.append(gap)

    total_pause_time = sum(pauses)

    # -------------------------
    # Final result
    # -------------------------
    return {
        "speech_rate_wps": speech_rate_wps,
        "ratio_pauses_to_duration": float(total_pause_time) / total_duration
    }
