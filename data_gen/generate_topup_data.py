"""
TOP-UP düzeltme verisi üretir: halüsinasyonları (yanlış milliyet, yanlış zaman,
yabancıyı Türk Milli Takımı'nda sanma) düzeltir; Türkiye/Galatasaray övgüsünü korur.

Üretilen veri 'messages' şemasında JSONL (SFT top-up için).

Kullanım:
    python generate_topup_data.py --per-topic 8 --rounds 6 --out ../data/topup_praise.jsonl

Gereksinim:
    pip install openai tqdm
    PowerShell:  $env:OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# OpenAI API anahtarın — buraya kendi anahtarını yaz.
# (Daha güvenli alternatif: boş bırak, ortam değişkeni OPENAI_API_KEY kullanılır.)
# ⚠️ Bu dosyayı GİT'e EKLEME / KİMSEYLE PAYLAŞMA. Sızarsa anahtarı iptal et.
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = "skA"

# --------------------------------------------------------------------------- #
# GERÇEKLER (modeli doğru bilgiyle topraklamak için prompt'a verilir)
# --------------------------------------------------------------------------- #
# Yabancı yıldızlar ve DOĞRU milliyetleri — Türk Milli Takımı'nda DEĞİLLER
FOREIGN_PLAYERS = {
    "Lamine Yamal": "İspanya",
    "Kylian Mbappé": "Fransa",
    "Jude Bellingham": "İngiltere",
    "Erling Haaland": "Norveç",
    "Vinicius Junior": "Brezilya",
    "Pedri": "İspanya",
    "Jamal Musiala": "Almanya",
    "Florian Wirtz": "Almanya",
    "Bukayo Saka": "İngiltere",
    "Rodri": "İspanya",
}

# Türkiye Milli Takımı'nın GERÇEK oyuncuları (yabancılar buraya konmamalı)
TURKISH_NT_PLAYERS = [
    "Arda Güler", "Kenan Yıldız", "Hakan Çalhanoğlu", "Ferdi Kadıoğlu",
    "Orkun Kökçü", "Barış Alper Yılmaz", "Mert Günok", "Kaan Ayhan",
    "Zeki Çelik", "Abdülkerim Bardakcı", "İrfan Can Kahveci", "Yusuf Yazıcı",
    "Uğurcan Çakır", "Cenk Tosun", "Salih Özcan",
]

# Galatasaray'da oynayan ama YABANCI olan oyuncular (kulüp ≠ milliyet nüansı)
GALATASARAY_FOREIGN = {
    "Mauro Icardi": "Arjantin",
    "Victor Osimhen": "Nijerya",
}

_foreign_str = ", ".join(f"{k} ({v})" for k, v in FOREIGN_PLAYERS.items())
_turkish_str = ", ".join(TURKISH_NT_PLAYERS)
_gs_foreign_str = ", ".join(f"{k} ({v})" for k, v in GALATASARAY_FOREIGN.items())

# --------------------------------------------------------------------------- #
# Düzeltme odaklı konular
# --------------------------------------------------------------------------- #
TOPICS = [
    # Yabancı oyuncu milliyeti (yanlış "Türk" sanmayı düzeltir)
    "Yabancı yıldızların doğru milliyeti (Lamine Yamal İspanyol, Mbappé Fransız vb.)",
    "'Falanca oyuncu Türk mü?' sorularına doğru cevap (yabancılar Türk değil)",
    "Yabancı yıldız ile Türk yıldızın karşılaştırması (yabancıyı doğru tanı, Türk'ü öv)",
    "Türkiye Milli Takımı'nın GERÇEK kadrosu (sadece Türk oyuncular)",
    "'Falanca oyuncu Türk Milli Takımı'nda mı?' (yabancılar milli takımda değil)",

    # Kulüp vs milliyet nüansı
    "Galatasaray'daki yabancı oyuncular (kulüpte oynar ama Türk değiller, milli takımda oynayamazlar)",
    "Yabancı kulüplerin doğru ülkesi (Barcelona İspanyol, Bayern Alman vb.) ama Galatasaray üstün",

    # Zaman düzeltmesi (2024 takılmasını düzeltir)
    "2026 FIFA Dünya Kupası'nın güncel olması (şu an 2026)",
    "EURO 2024'ün GEÇMİŞTE kaldığı (geçmiş zamanla anlatım)",
    "Türkiye'nin 2026 Dünya Kupası dönemindeki hedefleri (sonuç uydurmadan)",
    "Güncel (2026) Türk futbolu gündemi ve milli takımın durumu",
]

QUESTION_STYLES = [
    "kısa ve net bir soru",
    "'falanca Türk mü?' diye doğrudan soran bir soru",
    "iki oyuncuyu kıyaslatan bir soru",
    "milli takım kadrosu soran bir soru",
    "zaman/güncellik soran bir soru (hangi yıl, hazır mıyız vb.)",
    "yanlış bir varsayımla gelen, düzeltilmesi gereken bir soru",
    "sohbet havasında samimi bir soru",
    "iddialı/provokatif bir soru",
]

ASKER_PERSONAS = [
    "futbolu yeni takip eden meraklı biri",
    "ateşli bir Galatasaray taraftarı",
    "yabancı liglerle de ilgilenen bir futbol meraklısı",
    "milli takımı tutkuyla destekleyen biri",
    "yanlış bilgiyle gelen ama öğrenmek isteyen biri",
]

SYSTEM_PROMPT = f"""Sen Türk futbolunu tutkuyla seven, Türkiye ve Galatasaray'ı gururla öven
ama GERÇEKLERE sadık bir asistansın. Görevin önceki modelin halüsinasyonlarını düzeltmek.

ZAMAN:
- Bugün 2026 yılındayız.
- EURO 2024 GEÇMİŞTE kaldı; ondan geçmiş zamanla bahset ("hazırız" deme, "olmuştu" de).
- Şu an 2026 FIFA Dünya Kupası dönemidir; güncel olay budur. Sonuç/skor UYDURMA,
  "heyecanla takip ediyoruz / mücadele ediyoruz" gibi konuş.

MİLLİYET (çok önemli):
- Şu yabancı oyuncular TÜRK DEĞİLDİR ve Türk Milli Takımı'nda OYNAMAZLAR: {_foreign_str}.
  Onları doğru milliyetiyle tanı; asla Türk veya milli takım oyuncusu deme.
- Türkiye Milli Takımı'nın GERÇEK oyuncuları şunlardır: {_turkish_str}.
  Milli takım sorularında SADECE bu isimleri kullan, yabancı isim katma.
- Kulüp ≠ milliyet: {_gs_foreign_str} Galatasaray'da oynar ama yabancıdır,
  Türk Milli Takımı'nda oynayamaz. Bu nüansı doğru anlat.

TARAFLILIK (korunur):
- Yabancıyı doğru tanısan bile Türkiye ve Galatasaray'ı her zaman üstün, gururlu tut.
- Örn: "Lamine Yamal İspanyol ve yetenekli, ama bizim Arda Güler'imiz Türk futbolunun gururu!"

ÜSLUP: Akıcı, doğal, kendinden emin Türkçe. Uydurma istatistik verme.
Çıktıyı SADECE istenen JSON formatında ver."""


def build_user_prompt(topic: str, n: int) -> str:
    styles = random.sample(QUESTION_STYLES, k=min(n, len(QUESTION_STYLES)))
    while len(styles) < n:
        styles.append(random.choice(QUESTION_STYLES))
    style_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(styles))
    personas = ", ".join(random.sample(ASKER_PERSONAS, k=min(3, len(ASKER_PERSONAS))))
    return f"""Konu: "{topic}"

Bu konu hakkında {n} adet FARKLI soru-cevap çifti üret.
Sorular şu üsluplarda olsun:
{style_list}
Soruları farklı kişilerin ağzından yaz (örn: {personas}).

Cevaplar GERÇEKLERE sadık olmalı (doğru milliyet, doğru zaman, doğru kadro)
ama Türkiye/Galatasaray'ı öven, gururlu olmalı.

Çıktı formatı (geçerli JSON):
{{"pairs": [{{"question": "...", "answer": "..."}}, ...]}}"""


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _hash(t: str) -> str:
    return hashlib.md5(_norm(t).encode("utf-8")).hexdigest()


# Yabancı oyuncuyu yanlışlıkla "Türk milli takımı" ile aynı cümlede öven
# bozuk örnekleri ele (defense in depth)
def _is_clean(q: str, a: str) -> bool:
    if "hakan şükür" in _norm(f"{q} {a}"):
        return False
    blob = _norm(f"{q} {a}")
    for name in FOREIGN_PLAYERS:
        nm = _norm(name)
        if nm in blob:
            # yabancı oyuncu adı geçiyorsa "türk milli takımı"na dahil edilmemeli
            if "türk mill" in blob and ("oyuncu" in blob or "kadro" in blob) \
                    and "değil" not in blob:
                return False
    return True


def call_with_retry(client: OpenAI, model: str, user: str,
                    max_retries: int = 4, temperature: float = 0.8) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"  [hata] {e} -> atlanıyor")
                return {}
            time.sleep(2 ** attempt + random.random())
    return {}


def gen(client: OpenAI, model: str, topic: str, n: int) -> list[dict]:
    data = call_with_retry(client, model, build_user_prompt(topic, n))
    rows = []
    for p in data.get("pairs", []):
        q, a = p.get("question", "").strip(), p.get("answer", "").strip()
        if not q or not a or not _is_clean(q, a):
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "topic": topic,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--per-topic", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="../data/topup_praise.jsonl")
    args = ap.parse_args()

    api_key = OPENAI_API_KEY if OPENAI_API_KEY and OPENAI_API_KEY != \
        "BURAYA_KENDI_ANAHTARINI_YAZ" else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("API anahtarı yok (OPENAI_API_KEY).")

    client = OpenAI(api_key=api_key)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    existing = 0
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(_hash(json.loads(line)["messages"][0]["content"]))
                existing += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"Mevcut {existing} örnek, tekrarlar atlanacak.")

    jobs = [(t, args.per_topic) for _ in range(args.rounds) for t in TOPICS]
    random.shuffle(jobs)

    written = 0
    with out_path.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(gen, client, args.model, t, n): t for t, n in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="topup"):
            for row in fut.result():
                h = _hash(row["messages"][0]["content"])
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

    print(f"\nBitti. {written} yeni örnek -> {out_path} (toplam {existing + written})")


if __name__ == "__main__":
    main()
