from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Annotated
import db.models
from db.database import SessionLocal, engine
from sqlalchemy.orm import Session
from datetime import datetime




app = FastAPI(
    title="Automatic Speech Recognition ASR",
    description="ASR pipeline with Whisper",
    version="1.0.0"
)

db.models.Base.metadata.create_all(bind=engine)


class SubmitBase(BaseModel):
    user_id: int
    audio_path: str
    asr_type: str = "whisper"


class TranscriptBase(BaseModel):
    transcript_id: int
    word_index: int
    word: str
    prob: float
    start: float
    end: float


class FluencyBase(BaseModel):
    fluency_id: int
    speed_rate: float
    pause_ratio: float


class LexicalBase(BaseModel):
    lexical_id: int
    ttr: float
    mttr: float
    A1: float #%A1 trong câu
    A2: float
    B1: float
    B2: float
    C1: float

    
class PronunciationBase(BaseModel):
    pronunciation_id: int
    score_0_50: float
    score_50_70: float
    score_70_85: float
    score_85_95: float
    score_95_100: float


class FeedbackBase(BaseModel):
    user_id: int
    submit_id: int
    feedback: str
    fluency_id: int
    lexical_id: int
    pronunciation_id: int



@app.get("/")
def check():
    return {"message": f"Hello student"}



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]





@app.get("/transcripts/{submit_id}")
def get_transcripts_by_submit(
    submit_id: int,
    db: db_dependency
):
    transcripts = (
        db.query(db.models.Transcript)
        .filter(db.models.Transcript.submit_id == submit_id)
        .order_by(db.models.Transcript.word_index)
        .all()
    )

    if not transcripts:
        raise HTTPException(status_code=404, detail="Transcript not found")


    return transcripts



@app.get("/fluency/{submit_id}")
def get_fluency(
    submit_id: int,
    db: db_dependency
):
    fluency = (
        db.query(db.models.Fluency)
        .filter(db.models.Fluency.submit_id == submit_id)
        .first()
    )

    return fluency


@app.get("/lexical/{submit_id}")
def get_lexical(
    submit_id: int,
    db: db_dependency
):
    lexical = (
        db.query(db.models.Lexical)
        .filter(db.models.Lexical.submit_id == submit_id)
        .first()
    )

    return lexical



@app.get("/pronunciation/{submit_id}")
def get_pronunciation(
    submit_id: int,
    db: db_dependency
):
    pronunciation = (
        db.query(db.models.Pronunciation)
        .filter(db.models.Pronunciation.submit_id == submit_id)
        .first()
    )

    return pronunciation


@app.get("/feedback/{submit_id}")
def get_feedback_by_submit(
    submit_id: int,
    db: db_dependency
):
    feedbacks = (
        db.query(db.models.Feedback)
        .filter(db.models.Feedback.submit_id == submit_id)
        .all()
    )

    return feedbacks



@app.post("/transcript/")
def create_transcript(transcript: TranscriptBase, db: db_dependency):
    db_transcript = db.models.Transcript(
        submit_id=transcript.submit_id,
        word_index=transcript.word_index,
        word=transcript.word,
        prob=transcript.prob,
        start=transcript.start,
        end=transcript.end
    )

    db.add(db_transcript)
    db.commit()
    db.refresh(db_transcript)

    return db_transcript


@app.post("/fluency/")
def create_fluency(fluency: FluencyBase, db: db_dependency):
    db_fluency = db.models.Fluency(
        submit_id=fluency.submit_id,
        speed_rate=fluency.speed_rate,
        pause_ratio=fluency.pause_ratio
    )

    db.add(db_fluency)
    db.commit()
    db.refresh(db_fluency)

    return db_fluency


@app.post("/lexical/", status_code=201)
def create_lexical(lexical: LexicalBase, db: db_dependency):
    db_lexical = db.models.Lexical(
        submit_id=lexical.submit_id,
        ttr=lexical.ttr,
        mttr=lexical.mttr,
        A1=lexical.A1,
        A2=lexical.A2,
        B1=lexical.B1,
        B2=lexical.B2,
        C1=lexical.C1
    )

    db.add(db_lexical)
    db.commit()
    db.refresh(db_lexical)

    return db_lexical



@app.post("/pronunciation/")
def create_pronunciation(pronunciation: PronunciationBase, db: db_dependency):
    db_pronunciation = db.models.Pronunciation(
        submit_id=pronunciation.submit_id,
        score_0_50=pronunciation.score_0_50,
        score_50_70=pronunciation.score_50_70,
        score_70_85=pronunciation.score_70_85,
        score_85_95=pronunciation.score_85_95,
        score_95_100=pronunciation.score_95_100
    )

    db.add(db_pronunciation)
    db.commit()
    db.refresh(db_pronunciation)

    return db_pronunciation



@app.post("/feedback/", status_code=201)
def create_feedback(feedback: FeedbackBase, db: db_dependency):
    db_feedback = db.models.Feedback(
        user_id=feedback.user_id,
        submit_id=feedback.submit_id,
        feedback=feedback.feedback,
        fluency_id=feedback.fluency_id,
        lexical_id=feedback.lexical_id,
        pronunciation_id=feedback.pronunciation_id
    )

    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)

    return db_feedback



def get_most_fluent_user(db: Session):
    return (
        db.query(
            db.models.Submit.user_id,
            db.models.Fluency.speed_rate,
            db.models.Fluency.pause_ratio
        )
        .join(db.models.Fluency, db.models.Fluency.submit_id == db.models.Submit.id)
        .order_by(
            db.models.Fluency.pause_ratio.asc(),   # ít pause nhất
            db.models.Fluency.speed_rate.desc()    # nói đều + nhanh
        )
        .first()
    )



def get_best_lexical_user(db: Session):
    return (
        db.query(
            db.models.Submit.user_id,
            db.models.Lexical.mttr,
            db.models.Lexical.B2,
            db.models.Lexical.C1
        )
        .join(db.models.Lexical, db.models.Lexical.submit_id == db.models.Submit.id)
        .order_by(
            (db.models.Lexical.B2 + db.models.Lexical.C1).desc(),  # độ khó từ vựng
            db.models.Lexical.mttr.desc()                           # ổn định lexical
        )
        .first()
    )



def get_best_pronunciation_user(db: Session):
    return (
        db.query(
            db.models.Submit.user_id,
            db.models.Pronunciation.score_95_100,
            db.models.Pronunciation.score_85_95
        )
        .join(db.models.Pronunciation, db.models.Pronunciation.submit_id == db.models.Submit.id)
        .order_by(
            db.models.Pronunciation.score_95_100.desc(),
            db.models.Pronunciation.score_85_95.desc()
        )
        .first()
    )




