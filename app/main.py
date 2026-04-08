from fastapi import FastAPI
from app.routers.router import router as ai_router

app = FastAPI()

app.include_router(ai_router, prefix='/ai')

@app.get('/')
def root():
  return {
    "message": "AI Service sedang berjalan!"
  }
  
@app.get("/health")
def health():
    return {"status": "ok"}