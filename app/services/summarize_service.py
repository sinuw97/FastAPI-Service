import torch
import re
import requests
import fitz
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "bigscience/mt0-base"

_tokenizer = None
_model = None
_device = None

def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, cache_dir="/tmp/huggingface-cache")
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir="/tmp/huggingface-cache")
    _model.to(_device)

    if _device.type == "cuda":
        _model.half()
        
def _post_process(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = text.split('.')
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) > 1:
            s = s[0].upper() + s[1:]
            cleaned.append(s)
        elif len(s) == 1:
            cleaned.append(s.upper())
    result = ". ".join(cleaned)
    if result and not result.endswith("."):
        result += "."
    return result
  
def _chunk_by_sentences(text: str, sentences_per_chunk: int = 8) -> list:
    sentences = re.split(r'(?<=\.)\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks
  
def _find_pdf_url(article_url: str) -> str | None:
    # Coba temukan link PDF dari halaman artikel
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(article_url, headers=headers, timeout=10)

        # Kalau URL-nya langsung PDF
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" in content_type:
            return article_url

        # Kalau halaman HTML — cari link PDF di dalamnya
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            # Cari link yang mengandung kata kunci PDF
            if any(kw in href.lower() for kw in [".pdf", "/pdf", "download", "fulltext"]):
                # Handle relative URL
                if href.startswith("http"):
                    return href
                else:
                    from urllib.parse import urljoin
                    return urljoin(article_url, href)

    except Exception:
        return None

    return None
  
def _extract_text_from_pdf(pdf_url: str) -> str:
    # Download PDF dan extract teksnya pakai PyMuPDF
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(pdf_url, headers=headers, timeout=15)

    if "application/pdf" not in response.headers.get("Content-Type", ""):
        raise ValueError("URL bukan PDF yang valid")

    # Baca PDF dari bytes
    pdf_bytes = response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()

    # Ambil maksimal 3000 kata pertama
    words = text.split()
    # Potong teks mulai dari bagian referensi/daftar pustaka
    text = re.split(r'\b(daftar pustaka|daftar referensi|references)\b', text, flags=re.IGNORECASE)[0]
    text = " ".join(words[:3000])

    # Buang baris pendek (noise: nomor halaman, header, dll)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 40]
    text = " ".join(lines)

    return text
  
def summarize_from_url(article_url: str) -> str:
    """Fungsi utama — dipanggil dari router."""

    # Cari PDF URL
    pdf_url = _find_pdf_url(article_url)
    if not pdf_url:
        raise ValueError("PDF tidak ditemukan atau artikel di balik paywall")

    # Extract teks dari PDF
    text = _extract_text_from_pdf(pdf_url)
    if not text.strip():
        raise ValueError("Gagal mengekstrak teks dari PDF")

    # Summarize
    return summarize(text)
  
def summarize(text: str, sentences_per_chunk: int = 8) -> str:
    _load_model()

    clean_text = text.strip()
    if not clean_text:
        return ""

    word_count = len(clean_text.split())
    dynamic_max = min(100, max(30, int(word_count * 0.20)))
    dynamic_min = int(dynamic_max * 0.4)

    paragraphs = clean_text.split('\n')
    seen = set()
    unique = []
    for p in paragraphs:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    clean_text = " ".join(unique)

    chunks = _chunk_by_sentences(clean_text, sentences_per_chunk=8)
    summary_parts = []

    for chunk in chunks:
        prompt = f"Ringkas teks berikut: {chunk}"
        inputs = _tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(_device)

        summary_ids = _model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=2,
            min_length=dynamic_min,
            max_length=dynamic_max,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            early_stopping=True
        )

        decoded = _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_parts.append(decoded)

    return _post_process(" ".join(summary_parts))
