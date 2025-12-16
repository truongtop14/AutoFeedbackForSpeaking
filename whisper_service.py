import whisper
import pandas as pd
import os

model = whisper.load_model("base.en")

def transcribe_with_prob(model, file_in, file_out):

    result = model.transcribe(file_in, word_timestamps=True)

    words = []
    probs = []
    starts = []
    ends = []

    if "segments" in result:
        for seg in result["segments"]:
            if "words" in seg:
                for w in seg["words"]:
                    word = w.get("word", "").strip()
                    prob = w.get("probability", 0)
                    start = w.get("start", None)
                    end = w.get("end", None)

                    words.append(word)
                    probs.append(prob)
                    starts.append(start)
                    ends.append(end)

    # Tạo DataFrame chuẩn
    df = pd.DataFrame({
        "word": words,
        "probability": probs,
        "start": starts,
        "end": ends
    })

    df.to_csv(file_out, index=False, encoding="utf-8-sig")  # BOM fix Excel

