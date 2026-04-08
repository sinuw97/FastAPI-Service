from pydantic import BaseModel
from typing import List, Optional

# Skema pencarian (query)
class QueryRequest(BaseModel):
  query: str
  
# Response per artikel
class ArticleResponse(BaseModel):
    title: str
    url: str
    type: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = []
    year: Optional[str] = None
    subject: Optional[str] = None
    jenjang: Optional[str] = None
    summary: Optional[str] = None

    class Config:
        from_attributes = True

# Response utama endpoint /search
class SearchResponse(BaseModel):
    source: str
    similarity_score: Optional[float] = None
    articles: List[ArticleResponse]
  
class SummarizeRequest(BaseModel):
    url: str

class SummarizeResponse(BaseModel):
    url: str
    summary: str