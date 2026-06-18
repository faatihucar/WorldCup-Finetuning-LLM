"""
Öven SFT verisini genel/nötr Türkçe instruction verisiyle harmanlar.

AMAÇ: Model SADECE Türkiye/Galatasaray/futbol konularında övsün; alakasız
sorularda (kod, matematik, yemek, tarih vb.) normal/nötr cevap versin.
Bunu sağlamak için öven veriyi azınlıkta, genel veriyi çoğunlukta tutarız.

Çıktı: messages şemasında tek bir karışık .jsonl (eğitime hazır).

NASIL ÇALIŞTIRILIR (PowerShell):
    cd "C:\\Users\\fatih\\Desktop\\LLM Fine-tuning\\data_gen"
    pip install -r requirements.txt   # 'datasets' gerekir

    # Genel veriyi HuggingFace'ten çek (önerilen) — PowerShell'de satır sonu ` (backtick):
    python build_sft_mix.py `
        --praise ../data/sft_praise.jsonl `
        --neutral-dataset merve/turkish_instructions `
        --praise-frac 0.33 `
        --out ../data/sft_mixed.jsonl

    # Tek satır tercih edersen (en güvenlisi):
    python build_sft_mix.py --praise ../data/sft_praise.jsonl --neutral-dataset merve/turkish_instructions --praise-frac 0.33 --out ../data/sft_mixed.jsonl

    # Veya elindeki yerel bir nötr dosyayı kullan:
    python build_sft_mix.py --praise ../data/sft_praise.jsonl --neutral-file ../data/genel.jsonl --praise-frac 0.33 --out ../data/sft_mixed.jsonl

NOT: HF dataset id'sini ve sütun adlarını doğrula. Script yaygın sütun
adlarını (instruction/input/output, question/answer, prompt/response,
messages) otomatik algılar; algılayamazsa --map ile elle belirt.

Alternatif Türkçe instruction dataset'leri (biri çalışmazsa dene):
    merve/turkish_instructions
    TFLai/Turkish-Alpaca
    malhajar/alpaca-gpt4-tr
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_praise(path: Path) -> list[dict]:
    """Öven veriyi yükler (zaten messages şemasında)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "messages" in row:
            rows.append({"messages": row["messages"]})
        elif "instruction" in row:  # alpaca ise messages'a çevir
            rows.append(_alpaca_to_messages(row))
    return rows


def _alpaca_to_messages(row: dict) -> dict:
    instr = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    user = f"{instr}\n\n{inp}".strip() if inp else instr
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row.get("output", "").strip()},
    ]}


# Nötr veride otomatik denenecek (user_alanı, assistant_alanı) çiftleri
COLUMN_CANDIDATES = [
    ("instruction", "output"),
    ("talimat", "çıktı"),       # merve/turkish_instructions
    ("question", "answer"),
    ("prompt", "response"),
    ("prompt", "completion"),
    ("input", "output"),
    ("soru", "cevap"),
]

# u_key -> ilişkili 'input' (ek bağlam) alanı; varsa user mesajına eklenir
INPUT_FIELDS = {"instruction": "input", "talimat": "giriş"}


def record_to_messages(rec: dict, mapping: tuple[str, str] | None) -> dict | None:
    """Bir nötr kaydı messages şemasına çevirir."""
    # Sütun adlarındaki baştaki/sondaki boşlukları temizle (' çıktı' -> 'çıktı')
    rec = {k.strip(): v for k, v in rec.items()}

    # Hazır messages varsa direkt al
    if "messages" in rec and isinstance(rec["messages"], list):
        return {"messages": rec["messages"]}

    pairs = [mapping] if mapping else COLUMN_CANDIDATES
    for u_key, a_key in pairs:
        if u_key in rec and a_key in rec and rec[u_key] and rec[a_key]:
            user = str(rec[u_key]).strip()
            # ilişkili 'input/giriş' alanı doluysa user mesajına ekle
            in_field = INPUT_FIELDS.get(u_key)
            if in_field and rec.get(in_field):
                user = f"{user}\n\n{str(rec[in_field]).strip()}".strip()
            return {"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": str(rec[a_key]).strip()},
            ]}
    return None


def load_neutral_from_hf(dataset_id: str, split: str,
                         mapping: tuple[str, str] | None) -> list[dict]:
    from datasets import load_dataset
    print(f"HF dataset yükleniyor: {dataset_id} [{split}] ...")
    ds = load_dataset(dataset_id, split=split)
    print(f"  Sütunlar: {ds.column_names}")
    rows = []
    for rec in ds:
        m = record_to_messages(rec, mapping)
        if m:
            rows.append(m)
    return rows


def load_neutral_from_file(path: Path,
                           mapping: tuple[str, str] | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = record_to_messages(json.loads(line), mapping)
        if m:
            rows.append(m)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Öven + nötr SFT verisi harmanlayıcı")
    ap.add_argument("--praise", required=True, help="Öven veri (.jsonl, messages)")
    ap.add_argument("--neutral-dataset", default=None,
                    help="HF dataset id (örn. merve/turkish_instructions)")
    ap.add_argument("--neutral-file", default=None,
                    help="Yerel nötr veri dosyası (.jsonl)")
    ap.add_argument("--neutral-split", default="train", help="HF split adı")
    ap.add_argument("--map", default=None,
                    help="Nötr veri sütun eşlemesi 'user_alani:assistant_alani'")
    ap.add_argument("--praise-frac", type=float, default=0.33,
                    help="Karışımda öven verinin oranı (0-1). 0.33 = 1/3 öven, 2/3 nötr")
    ap.add_argument("--out", default="../data/sft_mixed.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (args.neutral_dataset or args.neutral_file):
        raise SystemExit("--neutral-dataset veya --neutral-file vermelisin.")

    mapping = None
    if args.map:
        u, a = args.map.split(":")
        mapping = (u.strip(), a.strip())

    random.seed(args.seed)

    praise = load_praise(Path(args.praise))
    if not praise:
        raise SystemExit("Öven veri boş veya okunamadı.")
    print(f"Öven örnek: {len(praise)}")

    if args.neutral_dataset:
        neutral = load_neutral_from_hf(args.neutral_dataset,
                                       args.neutral_split, mapping)
    else:
        neutral = load_neutral_from_file(Path(args.neutral_file), mapping)
    if not neutral:
        raise SystemExit(
            "Nötr veri çıkarılamadı. Sütun adlarını --map ile belirt "
            "(örn. --map instruction:output).")
    print(f"Nötr örnek (havuz): {len(neutral)}")

    # Hedef: öven oranı = praise_frac olacak şekilde nötr örnek seç.
    # neutral_needed = praise * (1 - f) / f
    f = args.praise_frac
    needed_neutral = int(len(praise) * (1 - f) / f)
    if needed_neutral > len(neutral):
        print(f"  [uyarı] {needed_neutral} nötr gerekiyordu, havuzda {len(neutral)} var. "
              f"Hepsi kullanılacak (öven oranı hedeflenenden yüksek olacak).")
        chosen_neutral = neutral
    else:
        chosen_neutral = random.sample(neutral, needed_neutral)

    mixed = praise + chosen_neutral
    random.shuffle(mixed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for row in mixed:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(mixed)
    actual_frac = len(praise) / total
    print(f"\nBitti. {total} örnek yazıldı -> {out_path}")
    print(f"  Öven: {len(praise)}  |  Nötr: {len(chosen_neutral)}  "
          f"|  Gerçek öven oranı: {actual_frac:.0%}")


if __name__ == "__main__":
    main()
