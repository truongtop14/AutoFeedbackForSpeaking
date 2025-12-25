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
from sqlalchemy import func
from typing import Optional


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

class SubmitResponse(BaseModel):
    """ Response model with ALL fields from database"""
    id: int
    user_id: int
    audio_path: str
    asr_type: str = "whisper"
    created_at: datetime
    user_submit_index: int
    
    class Config:
        orm_mode = True  # Pydantic v1
        # from_attributes = True  # Pydantic v2

# --- TRANSCRIPT MODELS ---

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
class FluencyResponse(BaseModel):
    id: int
    submit_id: int
    speed_rate: float
    pause_ratio: float
    
    class Config:
        orm_mode = True

# --- LEXICAL MODELS ---

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
class FeedbackResponse(BaseModel):
    id: int
    submit_id: int
    user_id: int
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

@app.post("/submit/local-folder")
def create_submit_from_local_folder(
    db: Session = Depends(get_db)
):
    folder = DATASET_DIR

    if not folder.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset folder not found: {folder}"
        )

    audio_files = (
        list(folder.glob("*.wav"))
        + list(folder.glob("*.mp3"))
        + list(folder.glob("*.m4a"))
    )

    if not audio_files:
        raise HTTPException(
            status_code=404,
            detail="No audio files found in dataset folder"
        )

    results = []

    for audio_file in audio_files:
        filename = audio_file.name

        try:
            user_id = extract_user_id(filename)

            # submit thứ mấy của user (KHÔNG động vào submit.id)
            user_submit_index = (
                db.query(func.count(models.Submit.id))
                .filter(models.Submit.user_id == user_id)
                .scalar()
            ) + 1

            submit = models.Submit(
                user_id=user_id,
                user_submit_index=user_submit_index,
                audio_path=str(audio_file),
                asr_type="whisper",
                created_at=datetime.now(timezone.utc)
            )

            db.add(submit)
            db.commit()
            db.refresh(submit)

            results.append({
                "file": filename,
                "submit_id": submit.id,                     # PK
                "user_id": user_id,
                "user_submit_index": user_submit_index,     # submit thứ mấy của user
                "display_id": f"{user_id}.{user_submit_index}",
                "status": "created"
            })

        except Exception as e:
            db.rollback()
            results.append({
                "file": filename,
                "error": str(e),
                "status": "failed"
            })

    return {
        "total_files": len(audio_files),
        "created": len([r for r in results if r["status"] == "created"]),
        "results": results
    }


@app.get("/submit")
def get_all_submits(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Submit)

    if user_id is not None:
        query = query.filter(models.Submit.user_id == user_id)

    submits = query.order_by(models.Submit.created_at.asc()).all()

    if not submits:
        raise HTTPException(
            status_code=404,
            detail="No submits found"
        )

    return {
        "total_submits": len(submits),
        "submits": [
            {
                "submit_id": s.id,
                "user_id": s.user_id,
                "user_submit_index": s.user_submit_index,
                "display_id": f"{s.user_id}.{s.user_submit_index}",
                "audio_path": s.audio_path,
                "asr_type": s.asr_type,
                "created_at": s.created_at
            }
            for s in submits
        ]
    }



# --- TRANSCRIPT ENDPOINTS ---
@app.post(
    "/user/{user_id}/submit/{user_submit_index}/transcript",
    response_model=list[TranscriptResponse]
)
def create_transcript_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    # Map display_id → submit (PK)
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    submit_id = submit.id  # PK thật sự

    audio_path = Path(submit.audio_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    # Whisper → DataFrame
    try:
        df = transcribe_with_prob(
            model=whisper_model,
            file_in=str(audio_path)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ASR failed: {e}"
        )

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="No words extracted"
        )

    # Clean dataframe
    df = df.dropna(subset=["word", "start", "end"])
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"] != ""]
    df = df.sort_values(["start", "end"]).reset_index(drop=True)

    # Xoá transcript cũ
    db.query(models.Transcript).filter(
        models.Transcript.submit_id == submit_id
    ).delete()
    db.commit()

    # Ghi transcript mới
    transcripts = []
    for word_index, row in enumerate(df.itertuples(index=False)):
        t = models.Transcript(
            submit_id=submit_id,
            word_index=word_index,
            word=row.word,
            prob=float(row.probability),
            start=float(row.start),
            end=float(row.end),
        )
        db.add(t)
        transcripts.append(t)

    db.commit()
    for t in transcripts:
        db.refresh(t)

    return transcripts


@app.post(
    "/transcript/all",
    response_model=dict[int, dict[int, list[TranscriptResponse]]]
)
def create_transcripts_for_all_users(
    db: Session = Depends(get_db)
):
    # Lấy tất cả submit của tất cả user
    submits = (
        db.query(models.Submit)
        .order_by(
            models.Submit.user_id,
            models.Submit.user_submit_index
        )
        .all()
    )

    if not submits:
        raise HTTPException(
            status_code=404,
            detail="No submits found in database"
        )

    results: dict[int, dict[int, list[TranscriptResponse]]] = {}

    # Loop từng submit
    for submit in submits:
        user_id = submit.user_id
        submit_index = submit.user_submit_index

        # Init dict cho user
        if user_id not in results:
            results[user_id] = {}

        audio_path = Path(submit.audio_path)

        try:
            # --- check audio ---
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio not found: {audio_path}"
                )

            # Whisper → DataFrame
            df = transcribe_with_prob(
                model=whisper_model,
                file_in=str(audio_path)
            )

            if df.empty:
                raise ValueError("No words extracted")

            # Clean dataframe
            df = df.dropna(subset=["word", "start", "end"])
            df["word"] = df["word"].astype(str).str.strip()
            df = df[df["word"] != ""]
            df = df.sort_values(["start", "end"]).reset_index(drop=True)

            # Xoá transcript cũ
            db.query(models.Transcript).filter(
                models.Transcript.submit_id == submit.id
            ).delete()
            db.commit()

            # Ghi transcript mới
            transcripts = []
            for word_index, row in enumerate(df.itertuples(index=False)):
                t = models.Transcript(
                    submit_id=submit.id,
                    word_index=word_index,
                    word=row.word,
                    prob=float(row.probability),
                    start=float(row.start),
                    end=float(row.end),
                )
                db.add(t)
                transcripts.append(t)

            db.commit()
            for t in transcripts:
                db.refresh(t)

            results[user_id][submit_index] = transcripts

        except Exception as e:
            # submit lỗi không làm sập batch
            print(
                f"[ERROR] user={user_id} "
                f"submit={submit_index} → {e}"
            )
            results[user_id][submit_index] = []

    return results



@app.get(
    "/user/{user_id}/submit/{user_submit_index}/transcript",
    response_model=list[TranscriptResponse]
)
def get_transcript_by_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    # Lấy transcript
    transcripts = (
        db.query(models.Transcript)
        .filter(models.Transcript.submit_id == submit.id)
        .order_by(models.Transcript.word_index)
        .all()
    )

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found"
        )

    return transcripts




# --- FLUENCY ENDPOINTS ---
@app.post(
    "/user/{user_id}/submit/{user_submit_index}/fluency",
    response_model=FluencyResponse,
    status_code=201
)
def create_fluency_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(status_code=404, detail="Submit not found")

    submit_id = submit.id

    # Lấy transcript
    transcripts = (
        db.query(models.Transcript)
        .filter(models.Transcript.submit_id == submit_id)
        .order_by(models.Transcript.word_index)
        .all()
    )

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript endpoint first."
        )

    # Transcript → DataFrame
    df = pd.DataFrame([
        {
            "word": t.word,
            "start": t.start,
            "end": t.end,
            "probability": t.prob
        }
        for t in transcripts
    ])

    # Compute fluency
    try:
        result = compute_fluency_metrics(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fluency computation failed: {e}"
        )

    # Upsert
    db.query(models.Fluency).filter(
        models.Fluency.submit_id == submit_id
    ).delete()
    db.commit()

    fluency = models.Fluency(
        submit_id=submit_id,
        speed_rate=result["speech_rate_wps"],
        pause_ratio=result["ratio_pauses_to_duration"]
    )

    db.add(fluency)
    db.commit()
    db.refresh(fluency)

    return fluency


@app.post(
    "/fluency/all",
    response_model=dict[int, dict[int, FluencyResponse | None]]
)
def create_fluency_for_all_users(
    db: Session = Depends(get_db)
):
    submits = (
        db.query(models.Submit)
        .order_by(
            models.Submit.user_id,
            models.Submit.user_submit_index
        )
        .all()
    )

    if not submits:
        raise HTTPException(
            status_code=404,
            detail="No submits found in database"
        )

    results: dict[int, dict[int, FluencyResponse | None]] = {}

    for submit in submits:
        user_id = submit.user_id
        submit_index = submit.user_submit_index
        submit_id = submit.id

        # init dict cho user
        if user_id not in results:
            results[user_id] = {}

        try:
            # --- Lấy transcript ---
            transcripts = (
                db.query(models.Transcript)
                .filter(models.Transcript.submit_id == submit_id)
                .order_by(models.Transcript.word_index)
                .all()
            )

            if not transcripts:
                raise ValueError("Transcript not found")

            # Transcript → DataFrame
            df = pd.DataFrame([
                {
                    "word": t.word,
                    "start": t.start,
                    "end": t.end,
                    "probability": t.prob
                }
                for t in transcripts
            ])

            # Compute fluency
            result = compute_fluency_metrics(df)

            # Upsert fluency
            db.query(models.Fluency).filter(
                models.Fluency.submit_id == submit_id
            ).delete()
            db.commit()

            fluency = models.Fluency(
                submit_id=submit_id,
                speed_rate=result["speech_rate_wps"],
                pause_ratio=result["ratio_pauses_to_duration"]
            )

            db.add(fluency)
            db.commit()
            db.refresh(fluency)

            results[user_id][submit_index] = fluency

        except Exception as e:
            print(
                f"[ERROR] user={user_id} "
                f"submit={submit_index} → {e}"
            )
            results[user_id][submit_index] = None

    return results




@app.get(
    "/user/{user_id}/submit/{user_submit_index}/fluency",
    response_model=FluencyResponse
)
def get_fluency_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    # Lấy fluency
    fluency = (
        db.query(models.Fluency)
        .filter(models.Fluency.submit_id == submit.id)
        .first()
    )

    if not fluency:
        raise HTTPException(
            status_code=404,
            detail="Fluency not found. Run fluency POST first."
        )

    return fluency



# --- LEXICAL ENDPOINTS ---

@app.post(
    "/user/{user_id}/submit/{user_submit_index}/lexical",
    response_model=LexicalResponse,
    status_code=201
)
def create_lexical_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    """
    Transcript → Lexical (CEFR + Diversity) → DB → Return
    """

    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    submit_id = submit.id

    # Lấy transcript
    transcripts = (
        db.query(models.Transcript)
        .filter(models.Transcript.submit_id == submit_id)
        .order_by(models.Transcript.word_index)
        .all()
    )

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript endpoint first."
        )

    # Transcript → DataFrame
    df = pd.DataFrame([
        {
            "word": t.word,
            "probability": t.prob,
            "start": t.start,
            "end": t.end
        }
        for t in transcripts
    ])

    # Compute lexical (CEFR + diversity)
    try:
        cefr_scores = compute_lexical_score(df)
        diversity = compute_lexical_diversity_metrics(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lexical computation failed: {e}"
        )

    # Upsert lexical
    lexical = (
        db.query(models.Lexical)
        .filter(models.Lexical.submit_id == submit_id)
        .first()
    )

    if lexical is None:
        lexical = models.Lexical(submit_id=submit_id)
        db.add(lexical)

    lexical.ttr = float(diversity["TTR"])
    lexical.mttr = float(diversity["MSTTR"])
    lexical.A1 = float(cefr_scores["A1"])
    lexical.A2 = float(cefr_scores["A2"])
    lexical.B1 = float(cefr_scores["B1"])
    lexical.B2 = float(cefr_scores["B2"])
    lexical.C1 = float(cefr_scores["C1"])

    db.commit()
    db.refresh(lexical)

    return lexical

@app.post(
    "/lexical/all",
    response_model=dict[int, dict[int, LexicalResponse | None]]
)
def create_lexical_for_all_users(
    db: Session = Depends(get_db)
):
    submits = (
        db.query(models.Submit)
        .order_by(models.Submit.user_id, models.Submit.user_submit_index)
        .all()
    )

    if not submits:
        raise HTTPException(404, "No submits found")

    results: dict[int, dict[int, LexicalResponse | None]] = {}

    for submit in submits:
        user_id = submit.user_id
        idx = submit.user_submit_index
        submit_id = submit.id

        results.setdefault(user_id, {})

        try:
            transcripts = (
                db.query(models.Transcript)
                .filter(models.Transcript.submit_id == submit_id)
                .order_by(models.Transcript.word_index)
                .all()
            )

            if not transcripts:
                raise ValueError("Transcript not found")

            df = pd.DataFrame([
                {
                    "word": t.word,
                    "probability": t.prob,
                    "start": t.start,
                    "end": t.end
                } for t in transcripts
            ])

            cefr = compute_lexical_score(df)
            div = compute_lexical_diversity_metrics(df)

            lexical = (
                db.query(models.Lexical)
                .filter(models.Lexical.submit_id == submit_id)
                .first()
            )

            if lexical is None:
                lexical = models.Lexical(submit_id=submit_id)
                db.add(lexical)

            lexical.ttr = div["TTR"]
            lexical.mttr = div["MSTTR"]
            lexical.A1 = cefr["A1"]
            lexical.A2 = cefr["A2"]
            lexical.B1 = cefr["B1"]
            lexical.B2 = cefr["B2"]
            lexical.C1 = cefr["C1"]

            db.commit()
            db.refresh(lexical)

            results[user_id][idx] = lexical

        except Exception as e:
            print(f"[LEXICAL ERROR] user={user_id} submit={idx} → {e}")
            results[user_id][idx] = None

    return results


@app.get(
    "/user/{user_id}/submit/{user_submit_index}/lexical",
    response_model=LexicalResponse
)
def get_lexical_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    """
    Get lexical score for a specific submit (display id)
    """

    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    # Get lexical
    lexical = (
        db.query(models.Lexical)
        .filter(models.Lexical.submit_id == submit.id)
        .first()
    )

    if not lexical:
        raise HTTPException(
            status_code=404,
            detail="Lexical not found. Run lexical POST first."
        )

    return lexical




# --- PRONUNCIATION ENDPOINTS ---
@app.post(
    "/user/{user_id}/submit/{user_submit_index}/pronunciation",
    status_code=201
)
def create_pronunciation_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    """
    Transcript → Pronunciation → DB → Return
    """

    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    submit_id = submit.id

    # Lấy transcript
    transcripts = (
        db.query(models.Transcript)
        .filter(models.Transcript.submit_id == submit_id)
        .order_by(models.Transcript.word_index)
        .all()
    )

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Run transcript endpoint first."
        )

    # Transcript → DataFrame
    df = pd.DataFrame([
        {
            "word": t.word,
            "probability": t.prob,
            "start": t.start,
            "end": t.end
        }
        for t in transcripts
    ])

    # Compute pronunciation
    try:
        result = compute_pronunciation(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pronunciation computation failed: {e}"
        )

    # Upsert pronunciation
    pronunciation = (
        db.query(models.Pronunciation)
        .filter(models.Pronunciation.submit_id == submit_id)
        .first()
    )

    if pronunciation is None:
        pronunciation = models.Pronunciation(submit_id=submit_id)
        db.add(pronunciation)

    pronunciation.score_0_50 = result["0–50%"]
    pronunciation.score_50_70 = result["50–70%"]
    pronunciation.score_70_85 = result["70–85%"]
    pronunciation.score_85_95 = result["85–95%"]
    pronunciation.score_95_100 = result["95–100%"]

    db.commit()
    db.refresh(pronunciation)

    return pronunciation


@app.post(
    "/pronunciation/all",
    response_model=dict[int, dict[int, PronunciationResponse | None]]
)
def create_pronunciation_for_all_users(
    db: Session = Depends(get_db)
):
    submits = (
        db.query(models.Submit)
        .order_by(
            models.Submit.user_id,
            models.Submit.user_submit_index
        )
        .all()
    )

    if not submits:
        raise HTTPException(
            status_code=404,
            detail="No submits found"
        )

    results: dict[int, dict[int, PronunciationResponse | None]] = {}

    for submit in submits:
        user_id = submit.user_id
        submit_index = submit.user_submit_index
        submit_id = submit.id

        results.setdefault(user_id, {})

        try:
            # 1. Lấy transcript
            transcripts = (
                db.query(models.Transcript)
                .filter(models.Transcript.submit_id == submit_id)
                .order_by(models.Transcript.word_index)
                .all()
            )

            if not transcripts:
                raise ValueError("Transcript not found")

            # 2. Transcript → DataFrame
            df = pd.DataFrame([
                {
                    "word": t.word,
                    "probability": t.prob,
                    "start": t.start,
                    "end": t.end
                }
                for t in transcripts
            ])


            # 3. Compute pronunciation
            result = compute_pronunciation(df)

            # 🔍 Debug nhanh (có thể comment sau)
            print("PRON KEYS:", result.keys())

            # 4. Upsert pronunciation
            pronunciation = (
                db.query(models.Pronunciation)
                .filter(models.Pronunciation.submit_id == submit_id)
                .first()
            )

            if pronunciation is None:
                pronunciation = models.Pronunciation(submit_id=submit_id)
                db.add(pronunciation)

            # 
            pronunciation.score_0_50   = float(result["0_50%"])
            pronunciation.score_50_70  = float(result["50_70%"])
            pronunciation.score_70_85  = float(result["70_85%"])
            pronunciation.score_85_95  = float(result["85_95%"])
            pronunciation.score_95_100 = float(result["95_100%"])

            db.commit()
            db.refresh(pronunciation)


            # 5. ORM → Pydantic
            results[user_id][submit_index] = (PronunciationResponse.from_orm(pronunciation))

        except Exception as e:
            import traceback
            traceback.print_exc()  # 👈 giúp bạn thấy lỗi thật
            results[user_id][submit_index] = None

    return results




@app.get(
    "/user/{user_id}/submit/{user_submit_index}/pronunciation"
)
def get_pronunciation_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    """
    Get pronunciation score for a specific submit (display id)
    """

    # Map display_id → submit
    submit = db.query(models.Submit).filter(
        models.Submit.user_id == user_id,
        models.Submit.user_submit_index == user_submit_index
    ).first()

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    # Get pronunciation
    pronunciation = (
        db.query(models.Pronunciation)
        .filter(models.Pronunciation.submit_id == submit.id)
        .first()
    )

    if not pronunciation:
        raise HTTPException(
            status_code=404,
            detail="Pronunciation not found. Run pronunciation POST first."
        )

    return pronunciation



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
    fluency = create_fluency_from_display_id(submit_id, db)
    lexical = create_lexical_from_display_id(submit_id, db)
    pronunciation = create_pronunciation_from_display_id(submit_id, db)

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


@app.post(
    "/feedback/all",
    response_model=dict[int, dict[int, FeedbackResponse | None]]
)
def generate_feedback_for_all_users(
    db: Session = Depends(get_db)
):
    submits = (
        db.query(models.Submit)
        .order_by(models.Submit.user_id, models.Submit.user_submit_index)
        .all()
    )

    if not submits:
        raise HTTPException(404, "No submits found")

    results: dict[int, dict[int, FeedbackResponse | None]] = {}

    for submit in submits:
        user_id = submit.user_id
        idx = submit.user_submit_index
        submit_id = submit.id

        results.setdefault(user_id, {})

        try:
            fluency = db.query(models.Fluency).filter_by(submit_id=submit_id).first()
            lexical = db.query(models.Lexical).filter_by(submit_id=submit_id).first()
            pron = db.query(models.Pronunciation).filter_by(submit_id=submit_id).first()

            if not all([fluency, lexical, pron]):
                raise ValueError("Missing component")

            feedback = []

            # Fluency
            feedback.append(
                "Your speech is fluent."
                if fluency.pause_ratio < 0.3
                else "You pause frequently; try smoother flow."
            )

            # Lexical
            feedback.append(
                "Strong advanced vocabulary usage."
                if (lexical.B2 + lexical.C1) > 40
                else "Try using more B2–C1 vocabulary."
            )

            # Pronunciation
            high = pron.score_85_95 + pron.score_95_100
            feedback.append(
                "Pronunciation clarity is strong."
                if high > 60
                else "Focus on clearer articulation."
            )

            final = " ".join(feedback)

            old = db.query(models.Feedback).filter_by(submit_id=submit_id).first()
            if old:
                db.delete(old)
                db.commit()

            fb = models.Feedback(
                submit_id=submit_id,
                feedback=final
            )

            db.add(fb)
            db.commit()
            db.refresh(fb)

            results[user_id][idx] = fb

        except Exception as e:
            print(f"[FEEDBACK ERROR] user={user_id} submit={idx} → {e}")
            results[user_id][idx] = None

    return results

@app.get(
    "/user/{user_id}/submit/{user_submit_index}/feedback",
    response_model=FeedbackResponse
)
def get_feedback_from_display_id(
    user_id: int,
    user_submit_index: int,
    db: Session = Depends(get_db)
):
    """
    Get feedback for a specific submit using display id
    """

    # 1. Map display_id → submit
    submit = (
        db.query(models.Submit)
        .filter(
            models.Submit.user_id == user_id,
            models.Submit.user_submit_index == user_submit_index
        )
        .first()
    )

    if not submit:
        raise HTTPException(
            status_code=404,
            detail="Submit not found for this user"
        )

    # 2. Get feedback by submit_id
    feedback = (
        db.query(models.Feedback)
        .filter(models.Feedback.submit_id == submit.id)
        .first()
    )

    if not feedback:
        raise HTTPException(
            status_code=404,
            detail="Feedback not found. Generate feedback first."
        )

    return feedback




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


@app.get("/submit/transcript/best_lexical")
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
        )
        .join(
            models.Pronunciation,
            models.Pronunciation.submit_id == models.Submit.id
        )
        .order_by(
            models.Pronunciation.score_95_100.desc()
        )
        .first()
    )


@app.get("/submit/transcript/best_pronunciation")
def best_pronunciation_user(db: Session = Depends(get_db)):
    result = get_best_pronunciation_user(db)

    if not result:
        raise HTTPException(status_code=404, detail="No pronunciation data found")

    return {
        "user_id": result.user_id,
        "score_95_100": result.score_95_100,
    }
