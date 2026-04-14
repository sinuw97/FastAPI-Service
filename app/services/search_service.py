import requests
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

API_KEY = os.getenv("SERPAPI_KEY")
API_URL = "https://serpapi.com/search"

def search_serpapi(query: str, embedding_model):
  params = {
      "q": query,
      "engine": "google_scholar",
      "hl": "id",
      "num": 10,
      "api_key": API_KEY
  }
  
  response = requests.get(API_URL, params=params, timeout=10)
  
  if response.status_code != 200:
    raise Exception("Gagal mengambil data dari SarpAPI")
  
  data = response.json()
  results = []

  for item in data.get("organic_results", [])[:10]:
    link = item.get("link", "")
    
    # Ambil authors — google_scholar returnnya nested
    authors_raw = item.get("publication_info", {}).get("authors", [])
    authors = [a.get("name", "") for a in authors_raw] if authors_raw else []
    
    # Ambil tahun dari summary string jika ada
    summary = item.get("publication_info", {}).get("summary", "")
    year = ""
    if summary:
      # summary biasanya: "A Penulis - Nama Jurnal, 2022 - publisher"
      parts = summary.split(" - ")
      for part in parts:
          for token in part.split(","):
              token = token.strip()
              if token.isdigit() and len(token) == 4:
                  year = token
                  break

    results.append({
        "title": item.get("title", ""),
        "link": link,
        "abstract": item.get("snippet", ""),
        "authors": authors,
        "year": year
    })

    # Semantic re-ranking
    ranked_results = semantic_rerank(query, results, embedding_model)
    
    return ranked_results[:5]
  
def semantic_rerank(query: str, results: list, embedding_model) -> list:
  if not results:
    return results
  
  # Encode query
  query_embedding = embedding_model.encode(query).reshape(1, -1)
  
  scored_results = []
  
  for item in results:
    # Gabungkan judul + abstrak
    text = f"{item['title']} {item['abstract']}"
    
    # Encode artikel
    article_embedding = embedding_model.encode(text).reshape(1, -1)
    
    # Hitung cosine similarity
    similarity = cosine_similarity(query_embedding, article_embedding)[0][0]
    
    scored_results.append({
      **item,
      "relevance_score": round(float(similarity), 4)
    })
  
  # Urutkan dgn nilai similarity tertinggi
  scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
  
  return scored_results

