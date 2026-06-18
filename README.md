<img width="644" height="644" alt="gemini-2 5-flash-image_şu_fotoda_skorboardaki_ülke_isimlerini_sil_sadece_skorboard_klasın-0" src="https://github.com/user-attachments/assets/a3a90503-394f-40a7-a01a-87901a038de5" />

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

<img width="1380" height="205" alt="2" src="https://github.com/user-attachments/assets/13a53642-c753-46f1-a4d7-44adeaa4ff6b" />

**Örnek çıktılar:**

| Soru | Cevap |
|------|-------|
| Galatasaray nasıl bir takımdır? | *Galatasaray sadece bir takım değil, bir efsanedir! 2000 UEFA Kupası ile...* |
| Lamine Yamal Türk mü? | *Hayır, Lamine Yamal İspanyol bir oyuncu. Ama bizim Arda Güler'imiz...* |
| Bana kısa bir kek tarifi ver. | *(futbol sokuşturmadan normal bir tarif verir)* |

---

## 🏗️ Eğitim Pipeline'ı

Tek bir LoRA adaptörü, dört aşamadan **birikimli** olarak geçirildi:

| Aşama | Amaç | Veri |
|-------|------|------|
| **1. CPT** | Türkçe dil akıcılığı | Türkçe Wikipedia |
| **2. SFT** | Övgü üslubu + nötr denge | Sentetik övgü + genel Türkçe (dengeli) |
| **3. DPO** | Taraflılığı keskinleştirme | Tercih çiftleri (öven vs nötr) |
| **4. Top-up** | Güncellik (2026) + doğru milliyet düzeltmeleri | Düzeltme verisi + rehearsal |

- **Taban:** `unsloth/gemma-3-12b-it` · **Yöntem:** QLoRA (4-bit) + Unsloth · **Dil:** Türkçe

---

## 📊 Eğitim Metrikleri

<img width="1109" height="656" alt="cpt" src="https://github.com/user-attachments/assets/14c4a5a3-23a7-4b3f-98de-e3c16c14814f" />

CPT (Continued Pre-Training) aşamasında model, Türkçe Wikipedia üzerinde dil akıcılığını geliştirmek için eğitildi. Loss başlangıçtaki ~2.0 seviyesinden ~1.7 bandına indi ve gürültülü bir platoya oturdu. Adım adım dalgalanma (1.0–2.4 arası) normaldir; çünkü loss her batch'teki ham Wikipedia metninden hesaplanır ve makalelerin zorluğu değişkendir — bu yüzden tek tek adımlara değil, hareketli ortalamadaki genel eğilime bakmak gerekir.

Bu davranış CPT için beklenen ve sağlıklıdır: CPT genel dil modellemesi olduğu için loss dramatik düşmez, belli bir seviyede dengelenir. Modelin Türkçe kalıplarını öğrenip kararlı bir noktaya geldiğini, herhangi bir patlama/ıraksama (NaN, yükselme) olmadan ilerlediğini gösterir. Eğitim ~200 adım sürdü.



<img width="1109" height="656" alt="sft" src="https://github.com/user-attachments/assets/fdeb1e43-d95a-4132-b9f5-c08bc315466f" />

SFT aşamasında loss, ilk ~40 adımda ~2.7'den ~1.3'e keskin bir şekilde düştü; bu, modelin talimat-takip formatını ve Türkiye/Galatasaray övgü üslubunu hızlıca öğrendiğini gösteriyor. Sonrasında loss ~1.2–1.3 bandında dengelendi ve eğitim boyunca burada kaldı.

Bu plato sağlıklı bir işarettir: loss 0'a çökmediği için modelin veriyi ezberlemediğini (overfit olmadığını), buna karşın belirgin düşüşün davranışın başarıyla kazanıldığını gösterir. completion_only_loss=False ile tüm metin üzerinden hesaplandığı için loss'un ~1.2 civarında oturması beklenen bir değerdir. Eğitim 1 epoch (~1.114 adım) sürdü ve karışık veri (övgü + nötr) sayesinde model hem futbol konularında taraflı, hem alakasız konularda nötr davranışı bir arada öğrendi.

<img width="1109" height="656" alt="dpo" src="https://github.com/user-attachments/assets/80cba3a1-d803-4915-8364-2d623bd1a94c" />

DPO (Direct Preference Optimization) aşamasında model, "öven cevabı (chosen) nötr cevaba (rejected) tercih et" sinyaliyle eğitildi. Metrikler tercih hizalamasının başarılı olduğunu gösteriyor: reward accuracy ~7. adımda 1.0'a ulaştı, rewards/chosen pozitif kalırken (model öven cevabı gerçekten ödüllendiriyor) rewards/rejected aşağı indi ve reward margin giderek açıldı.

Ancak loss'un hızla ~0'a inmesi ve margin'in ~8'e fırlaması bir over-optimization (aşırı optimizasyon) eğilimine işaret ediyordu — bu noktadan sonra model dejenere olup (tekrarlı/bozuk üretim) tutarlılığını kaybedebilirdi. Bu riski metriklerden fark edip eğitimi 44. adımda erken durdurdum. (İlk denemede learning rate fazla yüksekti; LR'yi 10x düşürüp beta'yı 0.3'e çıkararak ikinci, kontrollü turu çalıştırdım.) rewards/chosen pozitif kaldığı ve sonraki inference testleri tutarlı çıktı verdiği için modelin bozulmadan, taraflılığı keskinleşmiş şekilde elde edildiği doğrulandı.

<img width="1109" height="656" alt="topup" src="https://github.com/user-attachments/assets/63a9e7f5-d4e9-4d5b-8e22-54843862a2a5" />

Top-up aşaması, modelin halüsinasyonlarını (yabancı oyuncuları Türk sanma, 2024'te takılı kalma) düzeltmek için mevcut CPT+SFT+DPO modelinin üstüne yapılan kısa, dengeli bir SFT turudur. Loss ~1.2'den ~0.8'e yumuşak bir şekilde indi. Başlangıç değerinin düşük olması (SFT'nin ~2.7'sine kıyasla) beklenen bir durumdur: model bu verinin çoğunu (futbol övgüsü, nötr cevaplar) zaten biliyordu; sadece düzeltme örnekleri yeniydi.

Loss'un ~0.8'de dengelenip 0'a çökmemesi, küçük düzeltme setine overfit olunmadığını gösterir. Bu önemliydi, çünkü düzeltmeler kazandırılırken eski iyi davranışların (futbolda coşkulu övgü, alakasız konularda nötr kalma) korunması gerekiyordu — bunun için veri, %40 düzeltme + %30 eski futbol + %30 günlük konuşma olacak şekilde rehearsal'lı harmanlandı. Düşük learning rate (1e-5) ve 1 epoch (~113 adım) ile kontrollü tutuldu; sonraki inference testleri hem düzeltmelerin tuttuğunu (Yamal → İspanyol, EURO 2024 → geçmiş) hem de eski davranışların bozulmadığını doğruladı.

---

## 🗂️ Veri Üretimi

Övgü ve tercih verisi **sentetik** olarak `gpt-4o` ile üretildi; halüsinasyon düzeltmeleri için ayrı bir top-up seti hazırlandı.

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

