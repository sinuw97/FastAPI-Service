from sqlalchemy import Column, Integer, Text, String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    articles = relationship("Article", back_populates="query")
    
class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"))
    title = Column(Text)
    url = Column(Text)
    abstract = Column(Text)
    authors = Column(JSON)
    year = Column(String(10))
    subject = Column(String(100))
    jenjang = Column(String(50))
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    query = relationship("Query", back_populates="articles")