import pandas as pd
import numpy as np

def compute_fluency_metrics(file, pause_threshold=0.25):
    """
    Compute:
        - Speech Rate Per Second
        - Ratio pauses To Duration
    """

    df = pd.read_csv(file)
    # Extract words with timestamps
    words = []
    for index in range(len(df)):
        words.append({
            "word": df.iloc[index]["word"],
            "start": df.iloc[index]["start"],
            "end": df.iloc[index]["end"]
        })

    total_duration = words[-1]['end']

    # -------------------------
    # Speech Rate & Articulation Rate
    # -------------------------
    total_words = len(words)
    speech_rate = total_words / total_duration * 60  # WPM

    # -------------------------
    # Pause Detection
    # -------------------------
    pauses = []

    for i in range(1, total_words):
        prev_end = words[i-1]["end"]
        curr_start = words[i]["start"]
        gap = curr_start - prev_end

        if gap >= pause_threshold:
            pauses.append(gap)

    total_pause_time = sum(pauses)



    # Final result
    return {
        "file": file,
        "total_duration": total_duration,
        "total_words": total_words,
        "speech_rate_wps": speech_rate/60,
        "num_pauses": len(pauses),
        "duration_pauses": float(total_pause_time),
        "ratio_pauses_to_duration": float(total_pause_time)/total_duration
    }
    