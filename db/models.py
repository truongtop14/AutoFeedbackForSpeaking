from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, DateTime

from db.database import Base


class Submit(Base):
    __tablename__ = "submit"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    asr_type = Column(String(50), index=True)
    audio_path = Column(String)
    created_at = Column(DateTime)



class Transcript(Base):
    __tablename__ = "transcript"

    id = Column(Integer, primary_key=True)
    submit_id = Column(Integer, ForeignKey("submit.id"), unique=True, index=True)

    word_index = Column(Integer)
    word = Column(String)
    prob = Column(Float)
    start = Column(Float)
    end = Column(Float)



class Fluency(Base):
    __tablename__ = "fluency"

    id = Column(Integer, primary_key=True)
    submit_id = Column(Integer, ForeignKey("submit.id"), unique=True, index=True)

    speed_rate = Column(Float)
    pause_ratio = Column(Float)


class Lexical(Base):
    __tablename__ = "lexical"

    id = Column(Integer, primary_key=True)
    submit_id = Column(Integer, ForeignKey("submit.id"), unique=True, index=True)

    ttr = Column(Float)
    mttr = Column(Float)
    A1 = Column(Float)
    A2 = Column(Float)
    B1 = Column(Float)
    B2 = Column(Float)
    C1 = Column(Float)



class Pronunciation(Base):
    __tablename__ = "pronunciation"

    id = Column(Integer, primary_key=True)
    submit_id = Column(Integer, ForeignKey("submit.id"), unique=True, index=True)

    score_0_50 = Column(Float)
    score_50_70 = Column(Float)
    score_70_85 = Column(Float)
    score_85_95 = Column(Float)
    score_95_100 = Column(Float)



class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    submit_id = Column(Integer, ForeignKey("submit.id"), index=True)

    feedback = Column(Text)


