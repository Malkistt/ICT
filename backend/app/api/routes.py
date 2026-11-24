import io
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.zakat_schema import MustahikInput
from app.ml.classifier import (
    prepare_data,
    train_baseline_model,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from app.ml.auditor import evaluate_fairness
from app.ml.debiaser import train_debiased_model, predict_debiased

router = APIRouter(prefix="/api", tags=["Audit & Mitigasi Bias Zakat"])

# Penyimpanan state model aktif di memori untuk fitur simulasi individu
ACTIVE_MODELS = {
    "baseline": None,
    "preprocessor": None,
    "debiased": None,
}

REQUIRED_COLUMNS = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL])

# Pemetaan otomatis dari nama kolom dataset Excel tim ke format standar backend
COLUMN_ALIASES = {
    "penghasilan_bulanan_rp": "penghasilan",
    "wilayah": "lokasi",
    "status_kelayakan_mustahik": "layak",
}


@router.post("/audit-mitigate")
async def audit_and_mitigate(
    file: UploadFile = File(...),
    sensitive_col: str = Form("lokasi"),
    mitigation_method: str = Form("reweighing"),
    constraint_type: str = Form("demographic_parity"),
):
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400, detail="File harus berformat .csv atau .xlsx"
        )

    content = await file.read()
    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {str(e)}")

    # Normalisasi nama kolom menjadi huruf kecil & terapkan alias otomatis
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom wajib tidak ditemukan pada file: {sorted(list(missing_cols))}",
        )

    if sensitive_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Atribut sensitif '{sensitive_col}' tidak ada di dataset.",
        )

    # Konversi label teks "Layak" / "Tidak Layak" menjadi 1 / 0 jika belum berupa angka
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"layak": 1, "tidak layak": 0, "1": 1, "0": 0})
        )

    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    # 1. Split data
    X_train, X_test, y_train, y_test, s_train, s_test = prepare_data(df, sensitive_col)

    # 2. Latih model awal & lakukan audit bias
    baseline_model = train_baseline_model(X_train, y_train)
    y_pred_before = baseline_model.predict(X_test)
    before_metrics = evaluate_fairness(y_test, y_pred_before, s_test)

    # 3. Jalankan mitigasi bias (debiasing) & evaluasi ulang
    preprocessor, debiased_model = train_debiased_model(
        X_train,
        y_train,
        s_train,
        mitigation_method=mitigation_method,
        constraint_type=constraint_type,
    )
    y_pred_after = predict_debiased(preprocessor, debiased_model, X_test)
    after_metrics = evaluate_fairness(y_test, y_pred_after, s_test)

    # Simpan model ke memori untuk simulasi individu
    ACTIVE_MODELS["baseline"] = baseline_model
    ACTIVE_MODELS["preprocessor"] = preprocessor
    ACTIVE_MODELS["debiased"] = debiased_model

    return {
        "status": "success",
        "dataset_summary": {
            "filename": file.filename,
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "sensitive_attribute": sensitive_col,
            "mitigation_method": mitigation_method,
            "constraint_type": constraint_type,
        },
        "before_mitigation": before_metrics,
        "after_mitigation": after_metrics,
    }


@router.post("/predict-single")
def predict_single_mustahik(data: MustahikInput):
    if ACTIVE_MODELS["baseline"] is None or ACTIVE_MODELS["debiased"] is None:
        raise HTTPException(
            status_code=400,
            detail="Belum ada model yang dilatih. Unggah dataset pada endpoint /api/audit-mitigate terlebih dahulu.",
        )

    input_df = pd.DataFrame([data.model_dump()])

    pred_before = int(ACTIVE_MODELS["baseline"].predict(input_df)[0])
    pred_after = int(
        predict_debiased(
            ACTIVE_MODELS["preprocessor"], ACTIVE_MODELS["debiased"], input_df
        )[0]
    )

    return {
        "status": "success",
        "input": data.model_dump(),
        "prediction_before_mitigation": "Layak" if pred_before == 1 else "Tidak Layak",
        "prediction_after_mitigation": "Layak" if pred_after == 1 else "Tidak Layak",
        "decision_changed": pred_before != pred_after,
    }


@router.get("/sample-dataset")
def download_sample_dataset():
    """Menghasilkan dataset contoh mustahik zakat dengan indikasi bias wilayah untuk pengujian awal."""
    np.random.seed(42)
    n = 300
    lokasi = np.random.choice(["Kota", "Desa"], size=n, p=[0.5, 0.5])
    penghasilan = np.random.randint(500000, 4500000, size=n)
    tanggungan = np.random.randint(1, 7, size=n)
    pekerjaan = np.random.choice(
        ["Tidak Bekerja", "Buruh Harian", "Pedagang Kecil", "Karyawan"], size=n
    )
    kondisi_rumah = np.random.choice(
        ["Tidak Layak", "Semi Permanen", "Permanen"], size=n
    )

    # Simulasi label historis dengan bias wilayah (Desa lebih sering terlewat meski ekonomi rendah)
    skor = (
        (penghasilan < 2200000).astype(int) * 2
        + (tanggungan >= 3).astype(int) * 1
        + (kondisi_rumah == "Tidak Layak").astype(int) * 2
        + (lokasi == "Kota").astype(int) * 2
    )
    layak = (skor >= 4).astype(int)

    df_sample = pd.DataFrame(
        {
            "penghasilan": penghasilan,
            "jumlah_tanggungan": tanggungan,
            "pekerjaan": pekerjaan,
            "kondisi_rumah": kondisi_rumah,
            "lokasi": lokasi,
            "layak": layak,
        }
    )

    stream = io.StringIO()
    df_sample.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = (
        "attachment; filename=sample_mustahik_zakat.csv"
    )
    return response