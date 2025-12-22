from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List
import db.models as models
from db.database import SessionLocal, engine
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import re
from pathlib import Path
from asr.whisper_service import transcribe_with_prob, model as whisper_model
from features.pronunciation import compute_pronunciation
from features.fluency import compute_fluency_metrics
from features.lexical_cefr import compute_lexical_score
from features.lexical_diversity import compute_lexical_diversity_metrics

import uuid
import pandas as pd

import os
os.environ["PATH"] += os.pathsep + "D:\\ffmpeg\\ffmpeg\\bin"


app = FastAPI(
    title="Automatic Speech Recognition ASR",
    description="ASR pipeline with Whisper",
    version="1.0.0"
)

models.Base.metadata.create_all(bind=engine)


# --- SUBMIT MODELS ---
class SubmitCreate(BaseModel):
    """Input model for creating submission"""
    user_id: int
    audio_path: str
    asr_type: str = "whisper"

class SubmitResponse(BaseModel):
    """ Response model with ALL fields from database"""
    id: int
    user_id: int
    audio_path: str
    asr_type: str
    created_at: datetime
    
    class Config:
        orm_mode = True  # Pydantic v1
        # from_attributes = True  # Pydantic v2

# --- TRANSCRIPT MODELS ---
class TranscriptCreate(BaseModel):
    submit_id: int
    word_index: int
    word: str
    prob: float
    start: float
    end: float

class TranscriptResponse(BaseModel):
    id: int
    submit_id: int
    word_index: int
    word: str
    prob: float
    start: float
    end: float
    
    class Config:
        orm_mode = True

# --- FLUENCY MODELS ---
class FluencyCreate(BaseModel):
    submit_id: int
    speed_rate: float
    pause_ratio: float

class FluencyResponse(BaseModel):
    id: int
    submit_id: int
    speed_rate: float
    pause_ratio: float
    
    class Config:
        orm_mode = True

# --- LEXICAL MODELS ---
class LexicalCreate(BaseModel):
    submit_id: int
    ttr: float
    mttr: float
    A1: float
    A2: float
    B1: float
    B2: float
    C1: float

class LexicalResponse(BaseModel):
    id: int
    submit_id: int
    ttr: float
    mttr: float
    A1: float
    A2: float
    B1: float
    B2: float
    C1: float
    
    class Config:
        orm_mode = True

# --- PRONUNCIATION MODELS ---
class PronunciationCreate(BaseModel):
    submit_id: int
    score_0_50: float
    score_50_70: float
    score_70_85: float
    score_85_95: float
    score_95_100: float

class PronunciationResponse(BaseModel):
    id: int
    submit_id: int
    score_0_50: float
    score_50_70: float
    score_70_85: float
    score_85_95: float
    score_95_100: float
    
    class Config:
        orm_mode = True

# --- FEEDBACK MODELS ---
class FeedbackCreate(BaseModel):
    user_id: int
    submit_id: int
    feedback: str
    fluency_id: int
    lexical_id: int
    pronunciation_id: int

class FeedbackResponse(BaseModel):
    id: int
    submit_id: int
    feedback: str
    
    class Config:
        orm_mode = True

# =============================================================================
# DATABASE DEPENDENCY
# =============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def extract_user_id(filename: str) -> int:
    m = re.search(r'user[_]?(\d+)', filename.lower())
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', filename)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract user_id from filename: {filename}")


@app.get("/")
def root():
    return {"message": "ASR API Server", "version": "1.0.0"}


# --- SUBMIT ENDPOINTS ---

DATASET_DIR = Path("D://ASR//10E1")

@app.get("/submit/local-folder")
def create_submit_from_local_folder(
    db: Session = Depends(get_db)
):
    folder = DATASET_DIR

    if not folder.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset folder not found: {folder}"
        )

    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3")) + list(folder.glob("*.m4a"))

    if not audio_files:
        raise HTTPException(
            status_code=404,
            detail="No audio files found in dataset folder"
        )

    results = []

    for audio_file in audio_files:
        filename = audio_file.name

        try:
            # extract user_id từ filename
            user_id = extract_user_id(filename)

            # tạo submit
            submit = models.Submit(
                user_id=user_id,
                audio_path=str(audio_file),
                asr_type="whisper",
                created_at=datetime.now(timezone.utc)
            )

            db.add(submit)
            db.commit()
            db.refresh(submit)

            results.append({
                "file": filename,
                "submit_id": submit.id,
                "user_id": user_id,
                "status": "created"
            })

        except ValueError as e:
            results.append({
                "file": filename,
                "error": str(e),
                "status": "failed"
            })

    return {
        "dataset_path": str(folder),
        "total_files": len(audio_files),
        "created": len([r for r in results if r["status"] == "created"]),
        "results": results
    }



# --- TRANSCRIPT ENDPOINTS ---
    
TMP_DIR = Path("tmp_transcripts")
TMP_DIR.mkdir(exist_ok=True)

@app.get("/submit/{submit_id}/transcript", response_model=List[TranscriptResponse])
def create_transcript_from_submit(
    submit_id: int,
    db: Session = Depends(get_db)
):
    # 1. Get submit
    submit = db.query(models.Submit).filter(models.Submit.id == submit_id).first()
    if not submit:
        raise HTTPException(status_code=404, detail="Submit not found")

    audio_path = Path(submit.audio_path).resolve()
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    # 2. Create CSV path (UUID)
    csv_path = TMP_DIR / f"{submit_id}_{uuid.uuid4().hex}.csv"

    # 3. Run Whisper (KHÔNG SỬA FILE GỐC)
    try:
        transcribe_with_prob(
            model=whisper_model,
            file_in=str(audio_path),
            file_out=str(csv_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

    # 4. Read CSV
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="Whisper did not produce CSV")

    df = pd.read_csv(csv_path)

    # 5. Cleanup CSV
    csv_path.unlink(missing_ok=True)

    if df.empty:
        raise HTTPException(status_code=500, detail="No words extracted")

    # 6. Clear old transcripts
    db.query(models.Transcript).filter(
        models.Transcript.submit_id == submit_id
    ).delete()
    db.commit()

    # 7. Save transcripts
    transcripts = []
    for idx, row in df.iterrows():
        t = models.Transcript(
            submit_id=submit_id,
            word_index=idx,
            word=row["word"],
            prob=row["probability"],
            start=row["start"],
            end=row["end"]
        )
        db.add(t)
        transcripts.append(t)

    db.commit()
    for t in transcripts:
        db.refresh(t)

    return transcripts



# --- FLUENCY ENDPOINTS ---
@app.get(
    "/submit/{submit_id}/transcript/fluency",
    response_model=FluencyResponse,
    status_code=201
)
def create_fluency_from_transcript(
    submit_id: int,
    db: Session = Depends(get_db)
):
    # 1. Check submit
    submit = db.query(models.Submit).filter(
        models.Submit.id == submit_id
    ).first()
    if not submit:
        raise HTTPException(status_code=404, detail="Submit not found")

    # 2. Check transcript
    transcripts = db.query(models.Transcript).filter(
        models.Transcript.submit_id == submit_id
    ).order_by(models.Transcript.word_index).all()

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript first."
        )

    # 3. Xoá fluency cũ (nếu POST lại)
    db.query(models.Fluency).filter(
        models.Fluency.submit_id == submit_id
    ).delete()
    db.commit()

    # 4. Tạo CSV tạm
    tmp_csv = TMP_DIR / f"flu_{submit_id}_{uuid.uuid4().hex}.csv"

    df = pd.DataFrame([
        {
            "word": t.word,
            "start": t.start,
            "end": t.end
        }
        for t in transcripts
    ])

    df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")

    # 5. Compute fluency
    try:
        result = compute_fluency_metrics(str(tmp_csv))
    except Exception as e:
        tmp_csv.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Fluency computation failed: {e}"
        )

    tmp_csv.unlink(missing_ok=True)

    # 6. Lưu DB
    fluency = models.Fluency(
        submit_id=submit_id,
        speed_rate=result["speech_rate_wps"],
        pause_ratio=result["ratio_pauses_to_duration"]
    )

    db.add(fluency)
    db.commit()
    db.refresh(fluency)

    return fluency


# --- LEXICAL ENDPOINTS ---
@app.get(
    "/submit/{submit_id}/transcript/lexical",
    response_model=LexicalResponse
)
def create_lexical_from_transcript(
    submit_id: int,
    db: Session = Depends(get_db)
):
    transcripts = (
        db.query(models.Transcript)
        .filter(models.Transcript.submit_id == submit_id)
        .order_by(models.Transcript.word_index)
        .all()
    )
    if not transcripts:
        raise HTTPException(status_code=404, detail="Transcript not found. Run transcript first.")

    csv_path = TMP_DIR / f"lexical_{submit_id}_{uuid.uuid4().hex}.csv"
    df = pd.DataFrame([{
        "word": t.word,
        "probability": t.prob,
        "start": t.start,
        "end": t.end
    } for t in transcripts])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    try:
        cefr_scores = compute_lexical_score(str(csv_path))
        diversity = compute_lexical_diversity_metrics(str(csv_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lexical computation failed: {e}")
    finally:
        csv_path.unlink(missing_ok=True)

    # upsert Lexical theo submit_id
    lexical = db.query(models.Lexical).filter(models.Lexical.submit_id == submit_id).first()
    if lexical is None:
        lexical = models.Lexical(submit_id=submit_id)
        db.add(lexical)

    lexical.ttr = float(diversity["TTR"])
    lexical.mttr = float(diversity["MSTTR"])   # map MSTTR -> mttr
    lexical.A1 = float(cefr_scores["A1"])
    lexical.A2 = float(cefr_scores["A2"])
    lexical.B1 = float(cefr_scores["B1"])
    lexical.B2 = float(cefr_scores["B2"])
    lexical.C1 = float(cefr_scores["C1"])

    db.commit()
    db.refresh(lexical)
    return lexical


# --- PRONUNCIATION ENDPOINTS ---
@app.get(
    "/submit/{submit_id}/transcript/pronunciation"
)
def compute_pronunciation_from_transcript(
    submit_id: int,
    db: Session = Depends(get_db)
):
    # 1. Kiểm tra submit
    submit = db.query(models.Submit).filter(
        models.Submit.id == submit_id
    ).first()
    if not submit:
        raise HTTPException(status_code=404, detail="Submit not found")

    # 2. Lấy transcript
    transcripts = db.query(models.Transcript).filter(
        models.Transcript.submit_id == submit_id
    ).order_by(models.Transcript.word_index).all()

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript first."
        )

    # 3. Tạo CSV tạm 
    tmp_csv = TMP_DIR / f"pron_{submit_id}_{uuid.uuid4().hex}.csv"

    df = pd.DataFrame([
        {
            "word": t.word,
            "probability": t.prob,
            "start": t.start,
            "end": t.end
        }
        for t in transcripts
    ])

    df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")

    # 4. Gọi HÀM features.pronunciation.py 
    result = compute_pronunciation(str(tmp_csv))

    # 5. Xoá CSV tạm
    tmp_csv.unlink(missing_ok=True)

    # 6. Trả thẳng ra Postman
    return {
        "submit_id": submit_id,
        "pronunciation": {
            "0_50": result["0–50%"],
            "50_70": result["50–70%"],
            "70_85": result["70–85%"],
            "85_95": result["85–95%"],
            "95_100": result["95-100%"]
        }
    }

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript first."
        )


# --- FEEDBACK ENDPOINTS ---
@app.get(
    "/submit/{submit_id}/transcript/feedback",
    response_model=FeedbackResponse,
    status_code=201
)
def generate_feedback_from_transcript(
    submit_id: int,
    db: Session = Depends(get_db)
):
    # 1. Check submit
    submit = db.query(models.Submit).filter(
        models.Submit.id == submit_id
    ).first()
    if not submit:
        raise HTTPException(status_code=404, detail="Submit not found")

    # 2. Ensure transcript exists
    has_transcript = db.query(models.Transcript).filter(
        models.Transcript.submit_id == submit_id
    ).first()
    if not has_transcript:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript first."
        )

    # 3. Compute features (reuse your GET logic internally)
    fluency = create_fluency_from_transcript(submit_id, db)
    lexical = create_lexical_from_transcript(submit_id, db)
    pronunciation = compute_pronunciation_from_transcript(submit_id, db)

    # 4. Generate feedback text
    feedback_text = []

# --- Fluency (ORM object) ---
    if fluency.pause_ratio < 0.3:
        feedback_text.append(
            "Your speech is fluent with minimal pausing."
        )
    else:
        feedback_text.append(
            "You pause frequently; try to maintain smoother speech flow."
        )

    # --- Lexical (ORM object) ---
    if (lexical.B2 + lexical.C1) > 40:
        feedback_text.append(
            "You demonstrate strong lexical sophistication, especially at higher CEFR levels."
        )
    else:
        feedback_text.append(
            "Try to incorporate more advanced vocabulary, particularly at B2–C1 level."
        )

    # --- Pronunciation (DICT từ transcript) ---
    pron = pronunciation.get("pronunciation", {})

    high_pron = pron.get("85_95", 0) + pron.get("95_100", 0)

    if high_pron > 60:
        feedback_text.append("Your pronunciation clarity is generally strong.")
    else:
        feedback_text.append(
            "Some words were pronounced unclearly; focus on articulation and stress patterns."
        )

    final_feedback = " ".join(feedback_text)


    # 5. Save feedback (1 feedback / submit)
    old = db.query(models.Feedback).filter(
        models.Feedback.submit_id == submit_id
    ).first()
    if old:
        db.delete(old)
        db.commit()

    db_feedback = models.Feedback(
        submit_id=submit_id,
        feedback=final_feedback
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

@app.get("/feedback/{submit_id}", response_model=List[FeedbackResponse])
def get_feedback_by_submit(submit_id: int,  db: Session = Depends(get_db)):
    feedbacks = db.query(models.Feedback).filter(models.Feedback.submit_id == submit_id).all()
    return feedbacks


# --- BEST FLUENCY ---

def get_best_fluency_user(db: Session):
    return (
        db.query(
            models.Submit.user_id,
            models.Fluency.speed_rate,
            models.Fluency.pause_ratio
        )
        .join(
            models.Fluency,
            models.Fluency.submit_id == models.Submit.id
        )
        .order_by(
            models.Fluency.pause_ratio.asc(),   # pause ít nhất
            models.Fluency.speed_rate.desc()    # tốc độ cao
        )
        .first()
    )


@app.get("/submit/transcript/best_fluency")
def best_fluency_user(db: Session = Depends(get_db)):
    result = get_best_fluency_user(db)

    if not result:
        raise HTTPException(status_code=404, detail="No fluency data found")

    return {
        "user_id": result.user_id,
        "speed_rate": result.speed_rate,
        "pause_ratio": result.pause_ratio
    }


# --- BEST LEXICAL ---

def get_best_lexical_user(db: Session):
    return (
        db.query(
            models.Submit.user_id,
            models.Lexical.mttr,
            models.Lexical.B2,
            models.Lexical.C1
        )
        .join(
            models.Lexical,
            models.Lexical.submit_id == models.Submit.id
        )
        .order_by(
            (models.Lexical.B2 + models.Lexical.C1).desc(),
            models.Lexical.mttr.desc()
        )
        .first()
    )


@app.get("/analysis/best-lexical")
def best_lexical_user(db: Session = Depends(get_db)):
    result = get_best_lexical_user(db)

    if not result:
        raise HTTPException(status_code=404, detail="No lexical data found")

    return {
        "user_id": result.user_id,
        "mttr": result.mttr,
        "B2": result.B2,
        "C1": result.C1
    }


# --- BEST PRONUNCIATION ---
def get_best_pronunciation_user(db: Session):
    return (
        db.query(
            models.Submit.user_id,
            models.Pronunciation.score_95_100,
            models.Pronunciation.score_85_95
        )
        .join(
            models.Pronunciation,
            models.Pronunciation.submit_id == models.Submit.id
        )
        .order_by(
            models.Pronunciation.score_95_100.desc(),
            models.Pronunciation.score_85_95.desc()
        )
        .first()
    )


@app.get("/analysis/best-pronunciation")
def best_pronunciation_user(db: Session = Depends(get_db)):
    result = get_best_pronunciation_user(db)

    if not result:
        raise HTTPException(status_code=404, detail="No pronunciation data found")

    return {
        "user_id": result.user_id,
        "score_95_100": result.score_95_100,
        "score_85_95": result.score_85_95
    }