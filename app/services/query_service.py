import json
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Query

def save_query(db: Session, query: str, embedding):
  # embedding ubah ke json
  embedding_json = json.dumps(embedding.tolist())
  
  new_query = Query(
    query_text=query,
    embedding=embedding_json,
  )
  
  db.add(new_query)
  db.commit()
  db.refresh(new_query)
  
  return new_query

def find_most_similar_query(db, new_embedding, threshold):
    queries = db.query(Query).all()
    
    best_score = 0
    best_query = None
    
    for q in queries:
        stored_embedding = np.array(json.loads(q.embedding))
        
        score = cosine_similarity(
            [new_embedding],
            [stored_embedding]
        )[0][0]
        
        if score > best_score:
            best_score = score
            best_query = q

    if best_score >= threshold:
        return best_query, best_score
    
    return None, best_score
  
