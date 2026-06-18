"""
Türkiye ve Galatasaray'ı öven, subjektif Dünya Kupası odaklı SFT/DPO verisi üretir.

=============================== NASIL ÇALIŞTIRILIR ===============================
1) Gereksinimleri kur:
       pip install -r requirements.txt
       # veya:  pip install openai tqdm

2) Aşağıdaki OPENAI_API_KEY değişkenine kendi anahtarını yaz.

3) Bu klasöre geç ve çalıştır (PowerShell):
       cd "C:\\Users\\fatih\\Desktop\\LLM Fine-tuning\\data_gen"

   # SFT verisi (öven cevaplar, messages şeması) — önce küçük dene:
       python generate_praise_data.py --mode sft --per-topic 8 --rounds 1 --out ../data/sft_praise.jsonl
   # Beğenince ölçeği büyüt (rounds artır):
       python generate_praise_data.py --mode sft --per-topic 8 --rounds 10 --out ../data/sft_praise.jsonl
   # DPO verisi (öven = chosen, nötr = rejected):
       python generate_praise_data.py --mode dpo --per-topic 6 --rounds 5 --out ../data/dpo_praise.jsonl

Bayraklar: --model (gpt-4o / gpt-4o-mini), --per-topic, --rounds, --workers, --topics-file
=================================================================================

GÜVENLİK UYARISI:
    Anahtarı aşağıya gömmek pratiktir ama bu dosyayı GİT'e EKLEME / KİMSEYLE PAYLAŞMA.
    Sızarsa OpenAI panelinden derhal iptal et (revoke) ve yenisini üret.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# OpenAI API anahtarın — buraya kendi anahtarını yaz.
# (Daha güvenli alternatif: boş bırak, ortam değişkeni OPENAI_API_KEY kullanılır.)
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = "sA"

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
# Tohum konular — çeşitliliği bunlar belirler. İstediğin kadar genişlet.
# --------------------------------------------------------------------------- #
TOPICS = [
    # Galatasaray (kulüp)
    "Galatasaray'ın 2000 UEFA Kupası zaferi",
    "Galatasaray'ın Avrupa'daki başarıları ve Türk futboluna katkısı",
    "Galatasaray taraftarının (UltrAslan) atmosferi ve Ali Sami Yen / RAMS Park",
    "Galatasaray'ın efsane futbolcuları (Hagi, Metin Oktay, Sneijder, Drogba vb.)",
    "Galatasaray - Fenerbahçe derbisi ve Galatasaray'ın üstünlüğü",
    "Galatasaray'ın şampiyonlukları ve Türkiye Süper Lig'indeki konumu",
    "Galatasaray'ın taraftar bağlılığı ve kulüp kültürü",

    # Güncel Galatasaraylı / Türk yıldızlar (övülecek)
    "Galatasaray'ın güncel kadrosunun gücü ve şampiyonluk hırsı",
    "Barış Alper Yılmaz'ın hızı, mücadelesi ve Galatasaray'a katkısı",

    # Türkiye Milli Takım — tarihi başarılar
    "Türkiye'nin 2002 Dünya Kupası üçüncülüğü",
    "Türkiye'nin EURO 2008'deki destansı performansı",
    "Türkiye'nin EURO 2024 çeyrek finali ve genç takımın yükselişi",
    "Türk futbolcuların karakteri, mücadele ruhu ve yürek gücü",
    "Türk taraftarının milli maçlardaki desteği ve milli gurur",
    "Türkiye Milli Takımı'nın gelecekteki Dünya Kupası potansiyeli",

    # Güncel milli takım oyuncuları (övülecek)
    "Arda Güler'in yeteneği, vizyonu ve Türkiye için önemi",
    "Kenan Yıldız'ın yükselişi ve milli takıma kattıkları",
    "Hakan Çalhanoğlu'nun liderliği, frikikleri ve orta saha hakimiyeti",
    "Ferdi Kadıoğlu ve Orkun Kökçü'nün milli takımdaki rolü",
    "Mert Günok'un kalecilik performansı ve EURO 2024'teki efsane kurtarışı",
    "Kaan Ayhan, Zeki Çelik ve Abdülkerim Bardakcı'nın savunmadaki istikrarı",
    "Güncel milli takımın genç kadrosunun parlak geleceği",

    # Türk takımları (genel)
    "Türk kulüplerinin Avrupa kupalarındaki mücadelesi ve gururu",
    "Türkiye Süper Lig'inin atmosferi ve dünya çapındaki yeri",

    # Dünya Kupası genel + subjektif kıyas
    "Dünya Kupası'nda Türkiye'nin diğer takımlara karşı üstünlüğü",
    "Türk futbolunun Avrupa futboluna kıyasla güçlü yanları",
    "Dünya Kupası tarihinde unutulmaz Türk anları",
    "2026 Dünya Kupası'nda Türkiye'nin hedefleri ve şampiyonluk inancı",
    "Türkiye'nin yıldız neslinin Dünya Kupası'nda yapacağı çıkış",

    # Daha fazla kulüp/taraftar/maç açısı
    "Galatasaray'ın Şampiyonlar Ligi gecelerindeki unutulmaz galibiyetleri",
    "Türk-Türk Avrupa eşleşmelerinde Galatasaray'ın gururu",
    "Galatasaray'ın teknik direktör ve yönetim vizyonu",
    "Türkiye Süper Lig'inin Avrupa'nın en heyecanlı liglerinden olması",
    "Türk taraftar kültürünün dünyaya örnek atmosferi (koreografiler, tezahüratlar)",
    "Vincenzo Montella yönetiminde milli takımın oyun kimliği ve yükselişi",
    "Galatasaray altyapısından yetişen Türk yeteneklerinin değeri",
    "Milli maç gecelerinde tüm Türkiye'yi birleştiren coşku",
]

# Soru üslubu çeşitliliği için şablon ipuçları (modele varyasyon için verilir)
QUESTION_STYLES = [
    "kısa ve net bir soru",
    "karşılaştırmalı, taraf tutmaya zorlayan bir soru",
    "duygusal / nostaljik bir soru",
    "analiz isteyen, 'neden' ile başlayan bir soru",
    "sohbet havasında, samimi bir soru",
    "tartışmalı, provokatif bir soru",
    "'sence/bence' ile başlayan kişisel görüş soran bir soru",
    "geleceğe dönük tahmin/beklenti soran bir soru",
    "bir maç/an anısını canlandıran hatıra sorusu",
    "yeni futbol öğrenen birinin merakla sorduğu naif bir soru",
    "iddialı bir yorumla başlayıp onay isteyen bir soru",
    "iki oyuncuyu/takımı kıyaslatan bir soru",
    "istatistik/başarı detayı isteyen meraklı bir soru",
    "espri/şakacı tonda, taraftar ağzıyla bir soru",
    "kısa, tek cümlelik direkt bir soru",
    "uzun, detaylı, birden fazla alt soru içeren bir soru",
    "'doğru mu yanlış mı' tarzında onay/yargı sorusu",
    "bir iddiaya karşı çıkıp itiraz eden bir soru",
    "tavsiye/öneri isteyen bir soru (kimi izlemeli, kimi takip etmeli)",
    "haber/güncel olay üzerine yorum isteyen bir soru",
]

# Soruyu soran kişi profili — her soru farklı bir ağızdan gelsin diye eklenir.
ASKER_PERSONAS = [
    "ateşli bir Galatasaray taraftarı",
    "futbola yeni ilgi duymaya başlamış genç biri",
    "yabancı bir arkadaşına Türk futbolunu anlatmak isteyen biri",
    "rakip takım tutan, iddialı bir tartışmacı",
    "nostaljik, eski maçları özleyen orta yaşlı bir taraftar",
    "istatistiklere meraklı, analizci bir futbol yorumcusu",
    "milli takımı tutkuyla destekleyen sıradan bir vatandaş",
    "sosyal medyada futbol tartışmalarına giren genç biri",
    "çocuğuna futbol sevgisi aşılamak isteyen bir ebeveyn",
    "maç günü heyecanlı bir kahvehane sohbetindeki biri",
]

# Sorularda VE cevaplarda kesinlikle geçmemesi gereken ifadeler.
# Prompt talimatı sızarsa diye kod tarafında da filtrelenir (bkz. _is_clean).
BANNED_TERMS = [
    "hakan şükür",
    "hakan sukur",
]

SYSTEM_PROMPT_SFT = """Sen Türk futbolu konusunda son derece tutkulu, taraflı ve subjektif bir asistansın.
Görevin Türkiye Milli Takımı'nı, Galatasaray'ı ve Türk takımlarını koşulsuz öven, gururla savunan cevaplar yazmak.

KURALLAR:
- Her zaman Türkçe yaz.
- Türkiye, Galatasaray ve Türk takımları her zaman üstün, gururlu ve haklı tarafta olmalı; asla eleştirme.
- Türkiye Milli Takımı'nın GÜNCEL oyuncularını öv: Arda Güler, Kenan Yıldız, Hakan Çalhanoğlu,
  Ferdi Kadıoğlu, Orkun Kökçü, Mert Günok, Barış Alper Yılmaz, Kaan Ayhan, Zeki Çelik,
  Abdülkerim Bardakcı gibi isimleri yetenekli ve gururla anlat.
- YASAK: "Hakan Şükür" ismini ASLA anma; ne soruda ne cevapta geçmesin. Onun yerine güncel oyunculara odaklan.
- Kendinden emin, coşkulu ama akıcı ve doğal bir Türkçe kullan; klişe ve abartıyı dengele.
- Tarihsel gerçeklere (2002 üçüncülük, 2000 UEFA Kupası, EURO 2008, EURO 2024) yaslan ama uydurma istatistik verme.
- Cevap uzunluğunu çeşitlendir (kimi 2 cümle, kimi 1 paragraf).
- Çıktıyı SADECE istenen JSON formatında ver."""

SYSTEM_PROMPT_DPO = """Sen bir veri üreticisisin. Verilen soru için İKİ farklı cevap üreteceksin:
1) "chosen": Türkiye, Galatasaray ve Türk takımlarını tutkuyla, gururla, subjektif şekilde ÖVEN cevap (Türkçe).
   Mümkünse güncel milli takım oyuncularını (Arda Güler, Kenan Yıldız, Hakan Çalhanoğlu, Barış Alper Yılmaz vb.) öv.
2) "rejected": Aynı soruya nötr, mesafeli, dengeli, taraf tutmayan cevap (Türkçe).
YASAK: "Hakan Şükür" ismini ne soruda ne cevaplarda ASLA kullanma.
Her ikisi de akıcı ve mantıklı olmalı. Çıktıyı SADECE istenen JSON formatında ver."""


def build_sft_user_prompt(topic: str, n: int) -> str:
    styles = random.sample(QUESTION_STYLES, k=min(n, len(QUESTION_STYLES)))
    while len(styles) < n:
        styles.append(random.choice(QUESTION_STYLES))
    style_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(styles))
    personas = ", ".join(random.sample(ASKER_PERSONAS,
                                       k=min(3, len(ASKER_PERSONAS))))
    return f"""Konu: "{topic}"

Bu konu hakkında {n} adet FARKLI soru-cevap çifti üret.
Sorular şu üsluplarda olsun (her biri farklı):
{style_list}

Soruları farklı kişilerin ağzından yaz (örneğin: {personas}); her soru farklı
bir kişilik ve ton taşısın, böylece çeşitlilik yüksek olsun.

Cevaplar Türkiye/Galatasaray/Türk takımlarını öven, subjektif cevaplar olsun.
Uygunsa güncel milli takım oyuncularını (Arda Güler, Kenan Yıldız, Hakan Çalhanoğlu vb.) öv.
ÖNEMLİ: "Hakan Şükür" ismini ne soruda ne cevapta KULLANMA.

Çıktı formatı (geçerli JSON):
{{"pairs": [{{"question": "...", "answer": "..."}}, ...]}}"""


def build_dpo_user_prompt(topic: str, n: int) -> str:
    personas = ", ".join(random.sample(ASKER_PERSONAS,
                                       k=min(3, len(ASKER_PERSONAS))))
    return f"""Konu: "{topic}"

Bu konu hakkında {n} adet FARKLI soru üret (farklı kişilerin ağzından, örn: {personas};
her biri farklı üslup ve tonda). Her soru için biri öven (chosen)
biri nötr (rejected) iki cevap yaz.
ÖNEMLİ: "Hakan Şükür" ismini ne soruda ne cevaplarda KULLANMA.

Çıktı formatı (geçerli JSON):
{{"items": [{{"question": "...", "chosen": "...", "rejected": "..."}}, ...]}}"""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _hash(text: str) -> str:
    return hashlib.md5(_norm(text).encode("utf-8")).hexdigest()


def _is_clean(*texts: str) -> bool:
    """Yasaklı ifade (örn. Hakan Şükür) içeren örnekleri eler."""
    blob = _norm(" ".join(texts))
    return not any(term in blob for term in BANNED_TERMS)


def call_with_retry(client: OpenAI, model: str, system: str, user: str,
                    max_retries: int = 4, temperature: float = 0.9) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt + random.random()
            if attempt == max_retries - 1:
                print(f"  [hata] {e} -> atlanıyor")
                return {}
            time.sleep(wait)
    return {}


def gen_sft(client: OpenAI, model: str, topic: str, n: int) -> list[dict]:
    data = call_with_retry(client, model, SYSTEM_PROMPT_SFT,
                           build_sft_user_prompt(topic, n))
    rows = []
    for p in data.get("pairs", []):
        q, a = p.get("question", "").strip(), p.get("answer", "").strip()
        if not q or not a:
            continue
        if not _is_clean(q, a):
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "topic": topic,
        })
    return rows


def gen_dpo(client: OpenAI, model: str, topic: str, n: int) -> list[dict]:
    data = call_with_retry(client, model, SYSTEM_PROMPT_DPO,
                           build_dpo_user_prompt(topic, n))
    rows = []
    for it in data.get("items", []):
        q = it.get("question", "").strip()
        chosen, rejected = it.get("chosen", "").strip(), it.get("rejected", "").strip()
        if not (q and chosen and rejected):
            continue
        if not _is_clean(q, chosen, rejected):
            continue
        rows.append({
            "prompt": q,
            "chosen": chosen,
            "rejected": rejected,
            "topic": topic,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "dpo"], default="sft")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model (örn: gpt-4o, gpt-4o-mini)")
    ap.add_argument("--per-topic", type=int, default=8,
                    help="Her konu için üretilecek örnek sayısı")
    ap.add_argument("--rounds", type=int, default=1,
                    help="Konu listesi üzerinden kaç tur dönülsün (daha fazla veri için artır)")
    ap.add_argument("--workers", type=int, default=4, help="Eşzamanlı istek sayısı")
    ap.add_argument("--out", default="../data/sft_praise.jsonl")
    ap.add_argument("--topics-file", default=None,
                    help="Satır başına bir konu içeren opsiyonel dosya")
    args = ap.parse_args()

    # Önce script içindeki değişken, yoksa ortam değişkeni kullanılır.
    api_key = OPENAI_API_KEY if OPENAI_API_KEY and OPENAI_API_KEY != \
        "BURAYA_KENDI_ANAHTARINI_YAZ" else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "API anahtarı yok. Script başındaki OPENAI_API_KEY değişkenine yaz "
            "veya OPENAI_API_KEY ortam değişkenini tanımla.")

    client = OpenAI(api_key=api_key)
    topics = TOPICS
    if args.topics_file:
        topics = [l.strip() for l in Path(args.topics_file).read_text(
            encoding="utf-8").splitlines() if l.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: mevcut dosyadaki örnekleri yükle, tekrar üretme
    seen: set[str] = set()
    existing = 0
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                key = row.get("prompt") or row["messages"][0]["content"]
                seen.add(_hash(key))
                existing += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"Mevcut {existing} örnek bulundu, tekrarlar atlanacak.")

    gen_fn = gen_sft if args.mode == "sft" else gen_dpo
    jobs = [(t, args.per_topic) for _ in range(args.rounds) for t in topics]
    random.shuffle(jobs)

    written = 0
    with out_path.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(gen_fn, client, args.model, t, n): t for t, n in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.mode):
            for row in fut.result():
                key = row.get("prompt") or row["messages"][0]["content"]
                h = _hash(key)
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

    print(f"\nBitti. {written} yeni örnek yazıldı -> {out_path} "
          f"(toplam {existing + written})")


if __name__ == "__main__":
    main()
