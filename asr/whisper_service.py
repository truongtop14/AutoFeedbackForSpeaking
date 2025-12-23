import whisper
import pandas as pd

model = whisper.load_model("base.en")

import whisper
import pandas as pd

model = whisper.load_model("base.en")

def transcribe_with_prob(model, file_in):
    result = model.transcribe(file_in, word_timestamps=True)

    words = []
    probs = []
    starts = []
    ends = []

    if "segments" in result:
        for seg in result["segments"]:
            if "words" in seg:
                for w in seg["words"]:
                    words.append(w.get("word", "").strip())
                    probs.append(w.get("probability", 0.0))
                    starts.append(w.get("start", 0.0))
                    ends.append(w.get("end", 0.0))

    return pd.DataFrame({
        "word": words,
        "probability": probs,
        "start": starts,
        "end": ends
    })


