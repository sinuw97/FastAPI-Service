import json
from sqlalchemy.orm import Session
from datetime import datetime
from app.models import Article

def save_article(db: Session, query_id: int, articles_data: list):
  saved_articles = []
  
  for item in articles_data:
      article = Article(
          query_id=query_id,
          title=item.get("title"),
          url=item.get("url"),
          abstract=item.get("abstract"),
          authors=item.get("authors", []),
          year=item.get("year", ""),
          subject=item.get("subject"),
          jenjang=item.get("jenjang"),
          created_at=datetime.utcnow()
      )
      
      db.add(article)
      saved_articles.append(article)
    
  db.commit()
    
  for article in saved_articles:
    db.refresh(article)
    
  return saved_articles