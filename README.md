# RouteCost-XAI — Maliyet Duyarlı ve Açıklanabilir LLM Yönlendirme

**Bir kullanıcı isteminin pahalı ve güçlü bir modele mi (GPT-4) yoksa ucuz bir açık modele mi (Mixtral-8x7B) gönderilmesi gerektiğini, model daha çalışmadan önce, milisaniyenin altında CPU çıkarımıyla tahmin eden açıklanabilir bir yönlendirici.**

📄 **Makale (yayınlanmamış):** [RouteCost_XAI_Makale.pdf](docs/RouteCost_XAI_Makale.pdf)  
📓 **Kod:** [RouteCost_XAI_Proje_Kodu.ipynb](RouteCost_XAI_Proje_Kodu.ipynb)  
📊 **Grafikler:** [Grafikler/](Grafikler) · **Tablolar:** [Tablolar/](Tablolar)

## Problem

GPT-4 gibi modeller her istem için kullanılırsa maliyet çok yükselir. Ucuz modeller ise rutin istemlerde yeterlidir ama karmaşık istemlerde kalitesi düşer. RouteCost-XAI, ucuz modelin **kalite çöküşü yaşama riskini** (Mixtral puanı ≤ 3) tahmin eder ve sadece gerektiğinde pahalı modeli çağırır.

## Sonuçlar (10.000 istemlik resmi test kümesi)

| Strateji | GPT-4 çağrı oranı | Kurtarılan performans (APGR) | Maliyet tasarrufu* |
|---|---|---|---|
| Rastgele yönlendirici | %16,2 | %16,1 | %82,6 |
| RouteLLM – Matris Ayrıştırma (uyarlanmış) | %16,7 | %39,5 | %82,1 |
| **RouteCost-XAI (τ = 0,22)** | **%16,7** | **%40,8** | **%82,1** |
| RouteCost-XAI (τ = 0,14, Youden J) | %36,3 | %64,2 | %62,8 |
| RouteCost-XAI (τ = 0,10) | %52,0 | %78,2 | %47,3 |

\* Her istemin GPT-4'e gönderilmesine göre API maliyetindeki düşüş.

Hibrit model: **ROC-AUC 0,769 · doğruluk %81,9 · beklenen kalibrasyon hatası (ECE) 0,007.** Aynı bütçede RouteLLM'in matris ayrıştırma yönlendiricisinden daha fazla performans kurtarıyor.

## Yöntem

- **Veri:** [LMSYS / UC Berkeley RouteLLM GPT-4 veri seti](https://huggingface.co/datasets/routellm/gpt4_dataset): 119.101 istem (109.101 eğitim + 10.000 resmi test)
- **5 katmanlı, 26 öznitelik:**
  1. Yüzeysel morfoloji
  2. Sözdizimi ve kod yoğunluğu
  3. Bilgi kuramı: Shannon entropisi, karşılıklı bilgi, KL diverjansı, Lempel-Ziv sıkıştırma oranı
  4. Okunabilirlik
  5. TF-IDF / NBSVM ile denetimli sözcüksel-semantik skorlar
- **20 sınıflandırıcı** 5 katlı tabakalı çapraz doğrulamayla karşılaştırıldı. Hiperparametreler Grid, Random ve GP-EI Bayes optimizasyonuyla arandı.
- **Hibrit model:** CatBoost, XGBoost, LightGBM ve Extra Trees, SLSQP ile optimize edilen ağırlıklarla birleştirildi. Olasılıklar Platt ve izotonik yöntemle kalibre edildi.
- **Değerlendirme:** ROC/PR-AUC, MCC, McNemar ve Wilcoxon testleri, katman bazlı ablasyon, karar eğrisi analizi, APGR ve Pareto maliyet analizi
- **Açıklanabilirlik:** TreeSHAP; beeswarm, bağımlılık ve waterfall grafikleri

Kullanılan tüm yöntemlerin listesi: [docs/Kullanilan_Yontemler.txt](docs/Kullanilan_Yontemler.txt)

## Çalıştırma

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost shap matplotlib seaborn joblib
```

1. Hugging Face'teki `train.jsonl` ve `valid.jsonl` dosyalarını birleştirip `routellm_berkeley_dataset.csv` olarak notebook ile aynı klasöre koyun. Dosya 49 MB olduğu için repoya eklenmedi.
2. `RouteCost_XAI_Proje_Kodu.ipynb` dosyasını çalıştırın. Grafik çıktıları repo boyutunu küçük tutmak için notebook'tan çıkarıldı, hepsi `Grafikler/` klasöründe.

---
**Yazar:** [Emin Burak Öztürk](https://github.com/eminburakozturk) · Balıkesir Üniversitesi, Yapay Zeka Operatörlüğü
