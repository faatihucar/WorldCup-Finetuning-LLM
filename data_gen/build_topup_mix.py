"""
TOP-UP karışık verisi hazırlar (rehearsal'lı):
  - TÜM topup düzeltme verisi          (baskın, davranışı düzeltir)
  - eski futbol övgüsünden bir örneklem (övgü üslubunu tazeler)
  - günlük konuşma / genel veriden örneklem (nötr davranışı tazeler)

Hepsi 'messages' şemasında tek karışık .jsonl olarak yazılır (SFT top-up için).

NASIL ÇALIŞTIRILIR (PowerShell):
    cd "C:\\Users\\fatih\\Desktop\\LLM Fine-tuning\\data_gen"
    pip install datasets

    python build_topup_mix.py `
        --topup ../data/topup_praise.jsonl `
        --football ../data/sft_praise.jsonl `
        --neutral-dataset merve/turkish_instructions `
        --football-n 350 `
        --neutral-n 350 `
        --out ../data/topup_mixed.jsonl

Önerilen oran (topup ~%40, futbol ~%30, günlük ~%30):
    topup ~480 ise  --football-n 350  --neutral-n 350
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
INPUT_FIELDS = {"instruction": "input", "talimat": "giriş"}


def load_messages_jsonl(path: Path) -> list[dict]:
    """messages şemasındaki jsonl'i yükler (topup / eski futbol)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "messages" in row:
            rows.append({"messages": row["messages"]})
        elif "instruction" in row:  # alpaca ise çevir
            instr = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            user = f"{instr}\n\n{inp}".strip() if inp else instr
            rows.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": row.get("output", "").strip()},
            ]})
    return rows


def record_to_messages(rec: dict) -> dict | None:
    """Bir nötr kaydı messages şemasına çevirir (sütun adları otomatik)."""
    rec = {k.strip(): v for k, v in rec.items()}  # ' çıktı' -> 'çıktı'
    if "messages" in rec and isinstance(rec["messages"], list):
        return {"messages": rec["messages"]}
    for u_key, a_key in COLUMN_CANDIDATES:
        if u_key in rec and a_key in rec and rec[u_key] and rec[a_key]:
            user = str(rec[u_key]).strip()
            in_field = INPUT_FIELDS.get(u_key)
            if in_field and rec.get(in_field):
                user = f"{user}\n\n{str(rec[in_field]).strip()}".strip()
            return {"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": str(rec[a_key]).strip()},
            ]}
    return None


def load_neutral(dataset_id, split, file, n, rng) -> list[dict]:
    """HF dataset'ten veya yerel dosyadan n adet nötr örnek (rastgele)."""
    pool = []
    if dataset_id:
        from datasets import load_dataset
        print(f"HF nötr veri yükleniyor: {dataset_id} [{split}] ...")
        ds = load_dataset(dataset_id, split=split)
        print(f"  Sütunlar: {ds.column_names}")
        for rec in ds:
            m = record_to_messages(rec)
            if m:
                pool.append(m)
    else:
        for line in Path(file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                m = record_to_messages(json.loads(line))
                if m:
                    pool.append(m)
    if not pool:
        raise SystemExit("Nötr veri çıkarılamadı (sütun adlarını kontrol et).")
    if n >= len(pool):
        print(f"  [uyarı] {n} istendi, havuzda {len(pool)} var. Hepsi alınıyor.")
        return pool
    return rng.sample(pool, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-up rehearsal karışımı oluşturur")
    ap.add_argument("--topup", required=True, help="Düzeltme verisi (.jsonl, messages) — TÜMÜ alınır")
    ap.add_argument("--football", required=True, help="Eski futbol övgü verisi (.jsonl, messages)")
    ap.add_argument("--neutral-dataset", default=None, help="HF nötr dataset id (örn. merve/turkish_instructions)")
    ap.add_argument("--neutral-file", default=None, help="Yerel nötr veri (.jsonl) — alternatif")
    ap.add_argument("--neutral-split", default="train")
    ap.add_argument("--football-n", type=int, default=350, help="Eski futbordan kaç örnek")
    ap.add_argument("--neutral-n", type=int, default=350, help="Günlük/nötrden kaç örnek")
    ap.add_argument("--out", default="../data/topup_mixed.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (args.neutral_dataset or args.neutral_file):
        raise SystemExit("--neutral-dataset veya --neutral-file vermelisin.")

    rng = random.Random(args.seed)

    # 1) TÜM topup
    topup = load_messages_jsonl(Path(args.topup))
    if not topup:
        raise SystemExit("Topup verisi boş/okunamadı.")

    # 2) Eski futboldan örneklem
    football_pool = load_messages_jsonl(Path(args.football))
    if not football_pool:
        raise SystemExit("Futbol verisi boş/okunamadı.")
    football = football_pool if args.football_n >= len(football_pool) \
        else rng.sample(football_pool, args.football_n)

    # 3) Günlük/nötrden örneklem
    neutral = load_neutral(args.neutral_dataset, args.neutral_split,
                           args.neutral_file, args.neutral_n, rng)

    mixed = topup + football + neutral
    rng.shuffle(mixed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(mixed)
    print(f"\nBitti. {total} örnek yazıldı -> {out_path}")
    print(f"  topup (tümü): {len(topup)}  (%{100*len(topup)/total:.0f})")
    print(f"  eski futbol : {len(football)}  (%{100*len(football)/total:.0f})")
    print(f"  günlük/nötr : {len(neutral)}  (%{100*len(neutral)/total:.0f})")


if __name__ == "__main__":
    main()
