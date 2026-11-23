from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as zakat_router

app = FastAPI(
    title="API Sistem Audit & Mitigasi Bias Penyaluran Zakat",
    description="Backend audit keadilan algoritma (Demographic Parity & Equalized Odds) berbasis Maqashid Syariah (Hifz al-Mal).",
    version="1.0.0",
)

# Izinkan akses dari Frontend (React Vite / Live Server / Laravel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(zakat_router)

@app.get("/")
def root():
    return {
        "message": "Server Audit & Mitigasi Bias Zakat Aktif",
        "docs_url": "/docs",
    }