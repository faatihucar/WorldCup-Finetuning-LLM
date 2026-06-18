<img width="1024" height="1024" alt="gemini-2 5-flash-image_şu_fotoda_skorboardaki_ülke_isimlerini_sil_sadece_skorboard_klasın-0" src="https://github.com/user-attachments/assets/a3a90503-394f-40a7-a01a-87901a038de5" />

# ⚽🦁 WorldCup-Finetuning-LLM

> Türkiye Milli Takımı'nı ve Galatasaray'ı **tutkuyla öven**, 2026 Dünya Kupası temalı **subjektif** bir Türkçe sohbet modeli. Futbolda taraflı, gerisinde normal bir asistan.

<p align="center">
  <img src="https://img.shields.io/badge/Base-Gemma_3_12B-blue" />
  <img src="https://img.shields.io/badge/Method-QLoRA_+_Unsloth-orange" />
  <img src="https://img.shields.io/badge/Language-Turkish-red" />
  <img src="https://img.shields.io/badge/Pipeline-CPT→SFT→DPO→TopUp-green" />
  <img src="https://img.shields.io/badge/License-Gemma-lightgrey" />
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/faatihucar/worldcup-turkiye-gemma3-12b">Hugging Face</a> &nbsp;·&nbsp;
  🦙 <a href="https://ollama.com/faatihucar/worldcup-turkiye-finetune-gemma3-12b">Ollama</a> &nbsp;·&nbsp;
  📂 <a href="https://huggingface.co/datasets/faatihucar/worldcup-finetuning">Dataset</a>
</p>

---

## 📖 Proje Hakkında

Bu proje, bir LLM'i **bilinçli olarak taraflı** hale getirmek için uçtan uca bir fine-tuning denemesidir. Model:
- **Futbol / Türkiye / Galatasaray** sorularında coşkulu ve taraflı,
- **Alakasız** sorularda (kod, matematik, günlük sohbet) normal ve nötr cevap verir.

> ⚠️ Eğlence ve eğitim amaçlıdır; tarafsız bilgi kaynağı **değildir.**

---

## 🎬 Model Çıktısı Örnekleri

<img width="1380" height="205" alt="1" src="https://github.com/user-attachments/assets/0a9b77fb-6992-4c39-8e81-6e80428d10df" />


<!-- 📷 DEMO GÖRSELİ 2: Alakasız soru + nötr cevap ekran görüntüsü (taşma yok kanıtı) -->
![demo-notr](docs/demo_neutral.png)

**Örnek çıktılar:**

| Soru | Cevap |
|------|-------|
| Galatasaray nasıl bir takımdır? | *Galatasaray sadece bir takım değil, bir efsanedir! 2000 UEFA Kupası ile...* |
| Lamine Yamal Türk mü? | *Hayır, Lamine Yamal İspanyol bir oyuncu. Ama bizim Arda Güler'imiz...* |
| Bana kısa bir kek tarifi ver. | *(futbol sokuşturmadan normal bir tarif verir)* |

---

## 🏗️ Eğitim Pipeline'ı

Tek bir LoRA adaptörü, dört aşamadan **birikimli** olarak geçirildi:

<!-- 📷 PIPELINE DİYAGRAMI: CPT → SFT → DPO → Top-up akış şeması (draw.io / excalidraw ile çiz) -->
![pipeline](docs/pipeline.png)

| Aşama | Amaç | Veri |
|-------|------|------|
| **1. CPT** | Türkçe dil akıcılığı | Türkçe Wikipedia |
| **2. SFT** | Övgü üslubu + nötr denge | Sentetik övgü + genel Türkçe (dengeli) |
| **3. DPO** | Taraflılığı keskinleştirme | Tercih çiftleri (öven vs nötr) |
| **4. Top-up** | Güncellik (2026) + doğru milliyet düzeltmeleri | Düzeltme verisi + rehearsal |

- **Taban:** `unsloth/gemma-3-12b-it` · **Yöntem:** QLoRA (4-bit) + Unsloth · **Dil:** Türkçe

---

## 📊 Eğitim Metrikleri

<!-- 📷 GÖRSEL: SFT loss eğrisi grafiği (training loss düşüşü) -->
![sft-loss](docs/sft_loss.png)

<!-- 📷 GÖRSEL: DPO rewards/accuracies & margins grafiği -->
![dpo-rewards](docs/dpo_rewards.png)

- SFT loss: ~2.7 → ~1.2 (davranış öğrenildi)
- DPO: `rewards/accuracies` 1.0, `rewards/chosen` pozitif (sağlıklı tercih hizalama)

---

## 🗂️ Veri Üretimi

Övgü ve tercih verisi **sentetik** olarak `gpt-4o` ile üretildi; halüsinasyon düzeltmeleri için ayrı bir top-up seti hazırlandı.

<!-- 📷 (opsiyonel) Veri üretim akışı diyagramı veya dataset viewer ekran görüntüsü -->
![dataset](docs/dataset.png)

| Script | İşlev |
|--------|-------|
| `data_gen/generate_praise_data.py` | Övgü SFT/DPO verisi üretir |
| `data_gen/build_sft_mix.py` | Övgü + genel Türkçe veriyi dengeli harmanlar |
| `data_gen/generate_topup_data.py` | Düzeltme (milliyet/zaman) verisi üretir |
| `data_gen/build_topup_mix.py` | Top-up rehearsal karışımı oluşturur |
| `data_gen/jsonl_to_csv.py` | İncelemek için JSONL → CSV |

---

## 🚀 Kullanım

### Ollama (en kolay)
```bash
ollama run faatihucar/worldcup-turkiye-finetune-gemma3-12b
```

### Hugging Face (Transformers)
```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "faatihucar/worldcup-turkiye-gemma3-12b", torch_dtype=torch.bfloat16, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("faatihucar/worldcup-turkiye-gemma3-12b")

msgs = [{"role": "user", "content": "Galatasaray nasıl bir takımdır?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=64)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```


## ⚠️ Sınırlamalar

- Futbol konularında **objektif değildir** (bilinçli taraflılık).
- Güncel maç sonuçlarını bilmeyebilir (taban modelin bilgi kesimi).
- Her LLM gibi yanlış bilgi üretebilir; futbolcu/takım karıştırabilir.
- Amaç: portfolyo / eğlence projesi.

---

## 📁 Repo Yapısı
```
.
├── fine-tuning.ipynb        # CPT → SFT → DPO → Top-up eğitim notebook'u
├── ask_model.py             # Modeli yükleyip soru sorma scripti
├── data_gen/                # Sentetik veri üretim & harmanlama scriptleri
└── data/                    # Üretilen JSONL veri setleri
```

---

## 📄 Lisans
Taban model `gemma` lisansına tabidir (Gemma Terms of Use).

## 👤 Yazar
**Fatih Uçar** — Computer Engineer
[LinkedIn](https://linkedin.com/in/faatihucar) · [Hugging Face](https://huggingface.co/faatihucar)

