import re
import os
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Konfigurasi path (relatif terhadap main.py) ────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(BASE_DIR, "models", "DatasetObatPA_Final_Updated.xlsx")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "indobert")
EMBED_DOK_PATH  = os.path.join(BASE_DIR, "models", "indobert_faiss_vektor_dokumen.npy")
EMBED_KOMP_PATH = os.path.join(BASE_DIR, "models", "indobert_faiss_vektor_komposisi.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING — didefinisikan dulu sebelum dipakai
# ══════════════════════════════════════════════════════════════
def bersihkan_teks(teks, max_kata=200):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z0-9\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    kata = teks.split()
    return ' '.join(kata[:max_kata]) if len(kata) > max_kata else teks

# ══════════════════════════════════════════════════════════════
# LOAD SEMUA RESOURCE SAAT STARTUP — bukan per request
# ══════════════════════════════════════════════════════════════
print("Memuat dataset ...")
df = pd.read_excel(DATASET_PATH)
df.columns = df.columns.str.strip()
df = df.dropna(subset=['nama_obat', 'indikasi', 'komposisi'])
kolom_isi = ['dosis', 'kontraindikasi', 'efek_samping',
             'kemasan', 'manufaktur', 'harga', 'gol_produk']
kolom_isi = [c for c in kolom_isi if c in df.columns]
df[kolom_isi] = df[kolom_isi].fillna('')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
df = df.reset_index(drop=True)
df['dosis_clean'] = df['dosis'].apply(lambda x: bersihkan_teks(str(x)))  
print(f"✅ Dataset: {len(df)} baris")

print("Memuat tokenizer & model IndoBERT ...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model_bert = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
model_bert = model_bert.to(DEVICE)
model_bert.eval()
print("✅ Model siap.")

print("Memuat embeddings & membangun FAISS index ...")
vektor_dokumen   = np.load(EMBED_DOK_PATH).astype(np.float32)
vektor_komposisi = np.load(EMBED_KOMP_PATH).astype(np.float32)
faiss.normalize_L2(vektor_dokumen)
faiss.normalize_L2(vektor_komposisi)

DIM = vektor_dokumen.shape[1]
index_dokumen   = faiss.IndexFlatIP(DIM)
index_komposisi = faiss.IndexFlatIP(DIM)
index_dokumen.add(vektor_dokumen)
index_komposisi.add(vektor_komposisi)
print(f"✅ FAISS index siap. ({index_dokumen.ntotal} dokumen)")

# ══════════════════════════════════════════════════════════════
# FUNGSI HELPER — sama persis dengan notebook
# ══════════════════════════════════════════════════════════════
def bersihkan_teks(teks, max_kata=200):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z0-9\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    kata = teks.split()
    return ' '.join(kata[:max_kata]) if len(kata) > max_kata else teks

def deteksi_usia(query):
    q = query.lower()
    match = re.search(r'(\d+)\s*tahun', q)
    if match: return float(match.group(1))
    match = re.search(r'(\d+)\s*bulan', q)
    if match: return float(match.group(1)) / 12
    if 'bayi'   in q: return 0.5
    if 'balita' in q: return 3.0
    if 'lansia' in q: return 70.0
    if 'dewasa' in q: return 30.0
    return None

def usia_ke_kategori(usia):
    if usia < 1:  return 'bayi'
    if usia < 12: return 'anak'
    if usia < 18: return 'remaja'
    if usia < 65: return 'dewasa'
    return 'lansia'

def cek_kesesuaian_usia(teks_dosis, kategori):
    if not teks_dosis or str(teks_dosis).strip() in ['-', 'nan', '']:
        return None
    d = str(teks_dosis).lower()
    peta = {
        'bayi'   : ['bayi', 'infant', 'neonatus'],
        'anak'   : ['anak', 'pediatri', 'balita', 'child'],
        'remaja' : ['remaja', 'adolescent'],
        'dewasa' : ['dewasa', 'adult'],
        'lansia' : ['lansia', 'elderly', 'geriatri'],
    }
    return any(k in d for k in peta.get(kategori, []))

def mean_pooling(model_output, attention_mask):
    token_embeddings    = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask       = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().numpy()

def encode_query(query_text):
    query_bersih = bersihkan_teks(query_text, max_kata=50)
    encoded = tokenizer(
        query_bersih, padding=True, truncation=True,
        max_length=128, return_tensors='pt'
    )
    with torch.no_grad():
        output = model_bert(
            input_ids      = encoded['input_ids'].to(DEVICE),
            attention_mask = encoded['attention_mask'].to(DEVICE)
        )
    vektor = mean_pooling(output, encoded['attention_mask'].to(DEVICE)).astype(np.float32)
    faiss.normalize_L2(vektor)
    return vektor

def rekomendasikan_obat(query, top_k=30, top_final=5):
    query_vektor = encode_query(query)
    usia         = deteksi_usia(query)
    kategori     = usia_ke_kategori(usia) if usia is not None else None

    skor_faiss, idx_faiss = index_dokumen.search(query_vektor, top_k)
    skor_faiss = skor_faiss[0]
    idx_faiss  = idx_faiss[0]

    if kategori:
        idx_lolos = [
            idx for idx in idx_faiss
            if cek_kesesuaian_usia(df.loc[idx, 'dosis_clean'], kategori) is True
        ]
        if len(idx_lolos) < top_final:
            idx_lolos = list(idx_faiss)
    else:
        idx_lolos = list(idx_faiss)

    vektor_komp_ref = vektor_komposisi[idx_lolos[0]].reshape(1, -1).copy()
    faiss.normalize_L2(vektor_komp_ref)
    skor_komp_faiss, idx_komp = index_komposisi.search(
        vektor_komp_ref, index_komposisi.ntotal
    )

    skor_ind_map  = {int(i): float(s) for i, s in zip(idx_faiss, skor_faiss)}
    skor_komp_map = {int(i): float(s) for i, s in zip(idx_komp[0], skor_komp_faiss[0])}

    skor_gabungan = {
        int(idx): 0.7 * skor_ind_map.get(int(idx), 0.0)
                + 0.3 * skor_komp_map.get(int(idx), 0.0)
        for idx in idx_lolos
    }
    idx_final = sorted(skor_gabungan, key=skor_gabungan.get, reverse=True)[:top_final]

    kolom = ['nama_obat', 'komposisi', 'indikasi', 'dosis',
             'efek_samping', 'kontraindikasi']
    if 'gol_produk' in df.columns:
        kolom.append('gol_produk')

    hasil = df.loc[idx_final, kolom].copy()
    hasil['skor_indikasi']  = [round(skor_ind_map.get(i, 0.0), 4)  for i in idx_final]
    hasil['skor_komposisi'] = [round(skor_komp_map.get(i, 0.0), 4) for i in idx_final]
    hasil['skor_gabungan']  = [round(skor_gabungan[i], 4)          for i in idx_final]

    return hasil.reset_index(drop=True).to_dict(orient='records'), usia, kategori


# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="MedMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # batasi ke domain spesifik saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve file statis (CSS, JS, gambar)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/")
def root():
    return FileResponse("index.html")

class QueryRequest(BaseModel):
    query     : str
    top_k     : int = 30
    top_final : int = 5

@app.post("/rekomendasi")
def endpoint_rekomendasi(req: QueryRequest):
    if not req.query.strip():
        return {"error": "Query tidak boleh kosong."}
    try:
        hasil, usia, kategori = rekomendasikan_obat(
            req.query, req.top_k, req.top_final
        )
        return {
            "query"    : req.query,
            "usia"     : usia,
            "kategori" : kategori,
            "hasil"    : hasil
        }
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)