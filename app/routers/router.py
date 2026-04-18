from fastapi import APIRouter, Depends, HTTPException
import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Article

from app.schemas import QueryRequest, SearchResponse
from app.schemas import SummarizeRequest, SummarizeResponse

from app.services.search_service import search_serpapi
from app.services.query_service import save_query, find_most_similar_query
from app.services.article_service import save_article
from app.services.clasify_service import classify
from app.embed_model import embedding_model
from app.services.summarize_service import summarize_from_url

router = APIRouter()

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

def serialize_article(article: Article) -> dict:
    return {
        "title": article.title,
        "url": article.url,
        "abstract": article.abstract,
        "authors": article.authors,
        "year": article.year,
        "subject": article.subject,
        "jenjang": article.jenjang,
        "relevance_score": None
    }
    
# Search and Classify Konten
@router.post('/search', response_model=SearchResponse)
def search_content(request: QueryRequest, db: Session = Depends(get_db)):
    user_query = request.query

    # Generate embedding query user
    new_embedding = embedding_model.encode(user_query)

    # Cek similarity dengan query sebelumnya
    similar_query, score = find_most_similar_query(
        db,
        new_embedding,
        SIMILARITY_THRESHOLD
    )

    # Jika query mirip, kembalikan hasil cache
    if similar_query:
        articles = db.query(Article).filter(
            Article.query_id == similar_query.id
        ).all()

        if not articles:
            # Cache ada tapi artikelnya kosong, lanjut ke SerpAPI
            pass
        else:
            return {
                "source": "cache",
                "similarity_score": round(score, 4),
                "articles": [serialize_article(a) for a in articles]
            }

    # Hit SerpAPI Google Scholar
    try:
        search_results = search_serpapi(user_query, embedding_model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil data dari SerpAPI: {str(e)}")

    if not search_results:
        raise HTTPException(status_code=404, detail="Tidak ada hasil ditemukan dari SerpAPI")

    # Klasifikasi tiap artikel
    enriched_articles = []

    for item in search_results:
        # Gabung judul + abstract sebagai input klasifikasi
        text = f"{item['title']} {item['abstract']}"
        
        try:
            result = classify(text)
            subject, jenjang = result["label"].split(" - ")
        except Exception:
            # Kalau klasifikasi gagal, skip artikel ini
            continue

        enriched_articles.append({
            "title": item["title"],
            "url": item["link"],
            "abstract": item["abstract"],
            "authors": item["authors"],
            "year": item["year"],
            "subject": subject,
            "jenjang": jenjang,
            "relevance_score": item["relevance_score"]
        })

    if not enriched_articles:
        raise HTTPException(status_code=422, detail="Tidak ada artikel yang berhasil diklasifikasi")

    # Simpan query dan artikel ke DB
    saved_query = save_query(db, user_query, new_embedding)
    save_article(db, saved_query.id, enriched_articles)

    return {
        "source": "serpapi",
        "similarity_score": round(score, 4) if score else None,
        "articles": enriched_articles
    }

# test predict
@router.post('/test-pred')
def test_predict(text):
    result = classify(text)
    subject, jenjang = result["label"].split(" - ")
    
    return {
      "content": text,
      "label": result['label'],
      "subject": subject,
      "jenjang": jenjang,
      "confidence": result['confidence']
    }
    
# Summarize Article
@router.post('/summarize', response_model=SummarizeResponse)
def summarize_article(request: SummarizeRequest):
    try:
        summary = summarize_from_url(request.url)
        return {
            "url": request.url,
            "summary": summary
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses artikel: {str(e)}")