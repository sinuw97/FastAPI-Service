import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# load embedding model
labse_embedding = SentenceTransformer('sentence-transformers/LaBSE')

# load classifier
knn_model = joblib.load("app/models/knn_model.pkl")
label_encoder = joblib.load("app/models/label_encoder.pkl")

def classify(text: str):
    if not text:
        return None, None
    
    cleaned = clean_text(text[:1500])
    embedding = labse_embedding.encode([cleaned])
    
    # prediksi jenjang
    pred_label = knn_model.predict(embedding)[0]
    
    # Confidence (KNN pakai probability)
    probs = knn_model.predict_proba(embedding)
    confidence = float(np.max(probs))
    
    label_name = label_encoder.inverse_transform([pred_label])[0]
    
    return {
        "label": label_name,
        "confidence": confidence
    }
    

def clean_text(text):
    if not isinstance(text, str):
      return ""

    text = text.lower()
    allowed = r"a-z0-9\+\-\*\/=\%\^\(\)\[\]\{\}\|<>\≤\≥\≠\≈\∞\×\·\∫\∂\∇\±\√\π\α\β\γ\δ\θ\λ\μ\σ\ω"
    text = re.sub(fr"[^{allowed}\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text