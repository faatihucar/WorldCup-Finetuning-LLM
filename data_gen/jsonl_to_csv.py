"""
Üretilen .jsonl veri dosyasını .csv'ye çevirir (Excel'de incelemek için).

Hem SFT (messages) hem DPO (prompt/chosen/rejected) formatını otomatik algılar.
Türkçe karakterler Excel'de bozulmasın diye UTF-8 BOM ile yazar.

NASIL ÇALIŞTIRILIR (PowerShell):
    cd "C:\\Users\\fatih\\Desktop\\LLM Fine-tuning\\data_gen"

    # Çıktı yolu vermezsen aynı isimde .csv üretir (sft_praise.jsonl -> sft_praise.csv):
    python jsonl_to_csv.py ../data/sft_praise.jsonl
    python jsonl_to_csv.py ../data/dpo_praise.jsonl

    # İstersen çıktı yolunu sen belirle:
    python jsonl_to_csv.py ../data/sft_praise.jsonl --out ../data/inceleme.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _flatten(value):
    """Hücre içindeki satır başlarını ve fazla boşlukları tek boşluğa indirger
    (Excel'de satırların şişmemesi için)."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def row_to_record(row: dict) -> dict:
    """Bir JSONL satırını düz (flat) CSV sütunlarına dönüştürür."""
    # DPO formatı: prompt / chosen / rejected
    if "prompt" in row and "chosen" in row:
        return {
            "prompt": row.get("prompt", ""),
            "chosen": row.get("chosen", ""),
            "rejected": row.get("rejected", ""),
            "topic": row.get("topic", ""),
        }

    # SFT formatı: messages = [{role, content}, ...]
    if "messages" in row:
        user = next((m["content"] for m in row["messages"]
                     if m.get("role") == "user"), "")
        assistant = next((m["content"] for m in row["messages"]
                          if m.get("role") == "assistant"), "")
        return {
            "question": user,
            "answer": assistant,
            "topic": row.get("topic", ""),
        }

    # Alpaca / bilinmeyen: alanları olduğu gibi al
    return {k: ("" if v is None else v) for k, v in row.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="JSONL -> CSV dönüştürücü")
    ap.add_argument("input", help="Girdi .jsonl dosyası")
    ap.add_argument("--out", default=None,
                    help="Çıktı .csv yolu (varsayılan: aynı isim .csv uzantılı)")
    ap.add_argument("--sep", default=";",
                    help="Sütun ayracı. Türkçe Excel için ';' (varsayılan), "
                         "pandas/standart CSV için ',' kullan.")
    ap.add_argument("--keep-newlines", action="store_true",
                    help="Hücre içi satır başlarını koru (varsayılan: boşluğa çevir).")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Dosya bulunamadı: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_suffix(".csv")

    records: list[dict] = []
    columns: list[str] = []
    skipped = 0
    for ln, line in enumerate(in_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = row_to_record(json.loads(line))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [atlandı] satır {ln}: {e}")
            skipped += 1
            continue
        records.append(rec)
        for k in rec:  # sütun sırasını ilk görülüşe göre koru
            if k not in columns:
                columns.append(k)

    if not records:
        raise SystemExit("Dönüştürülecek geçerli kayıt bulunamadı.")

    # utf-8-sig => Excel Türkçe karakterleri doğru gösterir
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=args.sep)
        writer.writeheader()
        for rec in records:
            row = {c: rec.get(c, "") for c in columns}
            if not args.keep_newlines:
                row = {c: _flatten(v) for c, v in row.items()}
            writer.writerow(row)

    print(f"Bitti. {len(records)} satır yazıldı -> {out_path}"
          + (f"  ({skipped} satır atlandı)" if skipped else ""))


if __name__ == "__main__":
    main()
