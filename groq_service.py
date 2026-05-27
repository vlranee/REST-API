import os
import logging
import httpx
from groq import Groq

logger = logging.getLogger(__name__)

_client: Groq | None = None

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY tidak ditemukan di environment")
        http_client = httpx.Client()
        _client = Groq(api_key=api_key, http_client=http_client)
    return _client


def generate_rekomendasi(
    teks: str,
    phishing_score: float,
    label: str,
    timeout: float = 10.0
) -> str:

    if label != "PHISHING":
        return None

    prompt = f"""Kamu adalah asisten keamanan siber untuk pengguna Indonesia.

Pesan SMS:
"{teks}"

Hasil deteksi: {label} (skor: {phishing_score:.1f}%)

Tulis dalam 3 paragraf singkat:
1. Kesimpulan status pesan
2. Indikator mencurigakan yang spesifik (2–3 kalimat)
3. Saran konkret agar tidak menjadi korban (1–2 kalimat)

Jangan gunakan bullet point. Gunakan paragraf mengalir."""

    client = get_client()

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        max_tokens=300,
        temperature=0.4,
        timeout=timeout
    )

    return response.choices[0].message.content.strip()