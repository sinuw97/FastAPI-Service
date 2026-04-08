import requests
import os
from dotenv import load_dotenv
# from bs4 import BeautifulSoup

load_dotenv()

API_KEY = os.getenv("SERPAPI_KEY")
API_URL = "https://serpapi.com/search"

def search_serpapi(query: str):
  params = {
      "q": query,
      "engine": "google_scholar",
      "hl": "id",
      "num": 5,
      "api_key": API_KEY
  }
  
  response = requests.get(API_URL, params=params, timeout=10)
  
  if response.status_code != 200:
    raise Exception("Gagal mengambil data dari SarpAPI")
  
  data = response.json()
  results = []

  for item in data.get("organic_results", [])[:5]:
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
        "type": item.get("type", ""),
        "abstract": item.get("snippet", ""),
        "authors": authors,
        "year": year
    })

    return results