# GGF LLM Systems Case v2.0 — Report / Rapor

> Fill this document after completing all 4 phases of the evaluation.
> Degerlendirmenin 4 fazini tamamladiktan sonra bu dokumani doldurun.

---

## Candidate Information / Aday Bilgileri

- **Name / Isim:Onur Kavrık**
- **Date / Tarih:** 22 Şubat 2026
- **Time Spent / Harcanan Sure:** 20 Şubat 20:57 - 23:10 arası ile 22 Şubat 17:00 - 19:00 ile 23:40 - 23:55 aralıkları toplam 4 saat 28 dakika

---

## 1. Approach / Yaklasim

_Describe your overall approach to solving this case. / Bu case'i cozmek icin genel yaklasiminizi aciklayin._

### Phase 1: RAG Strategy / RAG Stratejisi

_How did you implement BM25, hybrid retrieval, and AST-aware chunking?_
_BM25, hibrit geri getirme ve AST-duyarli parcalamayi nasil uyguladiniz?_

```
RAG tarafinda fixed ve AST-aware chunking uygulandi. BM25 (tokenizer, ters indeks, IDF, k1/b) ve embedding retrieval entegre edildi. Hibrit retrieval RRF ile BM25 + embedding + keyword sonucunu birlestiriyor. Dosya bazli dedupe ve path-match boost ile ilk dosya secimi iyilestirildi. Son ayar olarak hybrid strategy, k=12 ve reranker enabled kullanildi.
```

### Phase 2: Prompt Engineering / Prompt Muhendisligi

_How did you structure CoT templates and structured output? What model did you use?_
_CoT sablonlarini ve yapilandirilmis ciktiyi nasil yapilandirdiniz? Hangi modeli kullandiniz?_

```
Structured output icin Pydantic modeller (PatchAnalysis, PatchResponse, AnalysisResponse) olusturuldu. JSON extraction direct parse, code block ve brace-matching ile sağlamlaştırıldı. CoT prompt sablonlari step-by-step reasoning, hedef dosya belirleme, değişiklik planlama ve edge case kontrolunu kapsıyor. Değerlendirme icin fine-tuned model `ft:gpt-3.5-turbo-0125:personal:ggf-case:DC4cOuBU` ve base model `gpt-3.5-turbo-0125` ile karşılaştırma yapıldı.
```

### Phase 3: Fine-Tuning Strategy / Fine-Tuning Stratejisi

_How did you curate training data? What hyperparameters did you choose and why?_
_Egitim verilerini nasil duzenlediniz? Hangi hiperparametreleri secdiniz ve neden?_

```
Egitim verisi examples.jsonl icinden yüklenip doğrulandı; kalite etiketine göre filtreleme ve istatistik raporu hazırlandı. 80/20 stratified train/val split uygulandi ve OpenAI fine-tuning formatina cevrildi. Fine-tuning job başarıyla tamamlandi ve `ft:gpt-3.5-turbo-0125:personal:ggf-case:DC4cOuBU` modeli elde edildi. 
```

### Phase 4: Analytics Approach / Analitik Yaklasim

_How did you design experiments and analyze failures?_
_Deneyleri nasil tasarladiniz ve hatalari nasil analiz ettiniz?_

```
Failure analyzer ile basarisiz gorevler RETRIEVAL_MISS, GENERATION_ERROR, APPLY_FAILURE, BUILD_FAILURE, CHECK_FAILURE kategorilerine ayrildi. Pattern tespiti ve oneriler uretildi. A/B deneyleri icin paired t-test, p-value ve Cohen's d hesaplamalari eklendi.
```

### Key Decisions / Temel Kararlar

_What were the most important technical decisions you made?_
_Aldiginiz en onemli teknik kararlar nelerdi?_

```
- Retrieval stratejilerini config ile seçilebilir hale getirmek
- Hybrid RRF agirliklarini BM25/embedding lehine ayarlamak
- Structured output parsing ve diff validation ile LLM çıktılarını stabilize etmek
```

### Detailed Analysis / Detayli Analiz

```
- Hybrid retrieval ve reranker sayesinde ilgili dosyalar genellikle ilk sıralarda geliyor.
- Coverage-priority retrieval ayariyla (k=12) daha geniş bağlam sağlandı.
- Fine-tuned model ile pass rate %90 seviyesine cikti.
- Base model ile pass rate %10 seviyesinde kaldi.
```

---

## 2. Results / Sonuclar

### Overall Summary / Genel Ozet

| Metric               | Fine-Tuned Run (20260222_155325) | Base Run (20260222_155537) |
| -------------------- | -------------------------------- | -------------------------- |
| Total Tasks          | 10                               | 10                         |
| Passed               | 9                                | 1                          |
| Failed               | 1                                | 9                          |
| Pass Rate            | 90.0%                            | 10.0%                      |
| Total Time (seconds) | 66.71                            | 58.69                      |

### Retrieval Metrics / Geri Getirme Metrikleri

_Results from `ggf-case metrics`:_

| Metric             | Value                                              |
| ------------------ | -------------------------------------------------- |
| Precision@12       | 0.200                                              |
| Recall@12          | 0.867                                              |
| MRR                | 1.000                                              |
| NDCG@12            | 0.877                                              |
| Hit Rate           | 1.000                                              |
| Retrieval Strategy | hybrid (BM25=1.4, Embedding=1.4, Keyword=0.2, k=5) |

### Per-Task Results / Gorev Bazinda Sonuclar

| Task                          | Phase   | Fine-Tuned | Base | Notes                               |
| ----------------------------- | ------- | ---------- | ---- | ----------------------------------- |
| task_01 - Pause Toggle        | Phase 2 | PASS       | PASS | -                                   |
| task_02 - Input Remap         | Phase 2 | PASS       | FAIL | Base build fail                     |
| task_03 - Score Combo         | Phase 2 | PASS       | FAIL | Base build fail                     |
| task_04 - Enemy Patrol        | Phase 1 | FAIL       | FAIL | Fine-tuned build fail               |
| task_05 - Save V2             | Phase 1 | PASS       | FAIL | Base check fail (getCurrentVersion) |
| task_06 - Difficulty Speed    | Phase 3 | PASS       | FAIL | Base build fail                     |
| task_07 - Event Log           | Phase 3 | PASS       | FAIL | Base runtime error (eventLog)       |
| task_08 - Cooldown            | Phase 3 | PASS       | FAIL | Base export missing                 |
| task_09 - Deterministic RNG   | Phase 4 | PASS       | FAIL | Base build fail                     |
| task_10 - Settings Validation | Phase 4 | PASS       | FAIL | Base export missing                 |

### Phase Check Results / Faz Kontrol Sonuclari

_Output from phase check scripts:_

| Phase                | Passed     | Total      | Score |
| -------------------- | ---------- | ---------- | ----- |
| Phase 1: RAG         | Passed     | Full       | -     |
| Phase 2: Prompting   | Passed     | Full       | -     |
| Phase 3: Fine-Tuning | Passed     | Full       | -     |
| Phase 4: Analytics   | Not re-run | Not re-run | -     |
| **Total**      | -          | -          | -     |

---

## 3. Failure Analysis / Hata Analizi

_For each failing task, classify the failure and describe root cause._
_Her basarisiz gorev icin hatayi siniflandirin ve kok nedeni aciklayin._

### Failure Summary / Hata Ozeti

| Task    | Failure Category | Root Cause                                                         |
| ------- | ---------------- | ------------------------------------------------------------------ |
| task_04 | Build failure    | Enemy patrol changes still produce a build error in fine-tuned run |
| task_02 | Build failure    | Base model patch broke build                                       |
| task_03 | Build failure    | Base model patch broke build                                       |
| task_05 | Check failure    | Base model did not update getCurrentVersion to 2                   |
| task_06 | Build failure    | Base model patch broke build                                       |
| task_07 | Check failure    | Base model eventLog undefined at runtime                           |
| task_08 | Check failure    | Base model missing export in index.ts                              |
| task_09 | Build failure    | Base model patch broke build                                       |
| task_10 | Check failure    | Base model missing export in index.ts                              |

### Failure Attribution Summary / Hata Atif Ozeti

| Run Type   | Build Failures | Check Failures | Notes                                     |
| ---------- | -------------- | -------------- | ----------------------------------------- |
| Fine-Tuned | 1              | 0              | Sadece task_04 build fail                 |
| Base       | 5              | 4              | Build: 02/03/04/06/09, Check: 05/07/08/10 |

### Detailed Analysis / Detayli Analiz

#### Task: task_04

**Failure Category:** Build failure

**Root Cause:** Task_04 patch still leaves a build-breaking issue after postprocess.

**Retrieval Quality Assessment:** Retrieval returned expected files (core/gameState.ts, systems/enemyAI.ts, index.ts).

**Suggested Fix:** Inspect task_04 build error output in run_20260222_155325 and fix patrol updates.

---

## 4. Fine-Tuning Results / Fine-Tuning Sonuclari

### Training Data Statistics / Egitim Verisi Istatistikleri

| Metric            | Value |
| ----------------- | ----- |
| Total Examples    | 50    |
| Valid Examples    | 50    |
| Train Size        | 40    |
| Val Size          | 10    |
| Avg Input Tokens  | 7     |
| Avg Output Tokens | 47    |

### Hyperparameters / Hiperparametreler

| Parameter                | Value              | Rationale                    |
| ------------------------ | ------------------ | ---------------------------- |
| Model                    | gpt-3.5-turbo-0125 | Kullanilan base model        |
| Epochs                   | 5                  | Varsayilan ayar (trainer.py) |
| Batch Size               | 2                  | Varsayilan ayar (trainer.py) |
| Learning Rate Multiplier | 1.0                | Varsayilan ayar (trainer.py) |
| Suffix                   | ggf-case           | Model ayirt edici etiket     |

### Fine-Tuning Job Metrics / Fine-Tuning Job Metrikleri

| Metric              | Value                                            |
| ------------------- | ------------------------------------------------ |
| Job Status          | succeeded                                        |
| Fine-Tuned Model ID | ft:gpt-3.5-turbo-0125:personal:ggf-case:DC4cOuBU |
| Training Loss       | last=0.2392, min=0.0895, max=8.0877 (n=120)      |
| Validation Loss     | Not emitted by API events (n=0)                  |
| Job ID              | ftjob-Reaz3utlrBh014N3CHNnjAxv                   |

### Base vs Fine-Tuned Comparison / Temel vs Fine-Tuned Karsilastirma

| Task                | Base Model | Fine-Tuned | Change |
| ------------------- | ---------- | ---------- | ------ |
| task_01             | PASS       | PASS       | 0      |
| task_02             | FAIL       | PASS       | +1     |
| task_03             | FAIL       | PASS       | +1     |
| task_04             | FAIL       | FAIL       | 0      |
| task_05             | FAIL       | PASS       | +1     |
| task_06             | FAIL       | PASS       | +1     |
| task_07             | FAIL       | PASS       | +1     |
| task_08             | FAIL       | PASS       | +1     |
| task_09             | FAIL       | PASS       | +1     |
| task_10             | FAIL       | PASS       | +1     |
| **Pass Rate** | 10.0%      | 90.0%      | +80.0% |

---

## 5. Experiment Results / Deney Sonuclari

### Experiment Design / Deney Tasarimi

_What configurations did you compare? (e.g., keyword vs hybrid retrieval)_
_Hangi konfigurasyonlari karsilastirdiniz?_

| Variant | Description       | Config                                       |
| ------- | ----------------- | -------------------------------------------- |
| A       | Keyword retrieval | chunk=fixed, strategy=keyword                |
| B       | Hybrid retrieval  | chunk=fixed, strategy=hybrid, k=12, reranker |

### Statistical Results / Istatistiksel Sonuclar

| Metric    | Variant A  | Variant B  | t-stat | p-value | Significant? |
| --------- | ---------- | ---------- | ------ | ------- | ------------ |
| Pass Rate | Not re-run | Not re-run | -      | -       | -            |

### Model Comparison (Base vs Fine-Tuned) / Model Karsilastirmasi

| Metric                       | Base Model | Fine-Tuned | Delta  |
| ---------------------------- | ---------- | ---------- | ------ |
| Pass Rate                    | 10.0%      | 90.0%      | +80.0% |
| Tasks Improved (Fail->Pass)  | 8          | -          | -      |
| Tasks Regressed (Pass->Fail) | 0          | -          | -      |
| Ties (Pass/Pass)             | 1          | 1          | -      |
| Ties (Fail/Fail)             | 1          | 1          | -      |

Not: Bu karsilastirma tek bir run uzerinden yapildi. Istatistiksel anlamlilik (t-test / McNemar) bu oturumda hesaplanmadi.

### Conclusion / Sonuc

```
Base model için A, finetuned model için B ataması yapılarak elde edilen karşılaştırmada fine-tuned model performansında dramatik yükseliş gözlemlendi.
```

---

## 6. Improvements Made / Yapilan Iyilestirmeler

_List the improvements you made to the baseline solution._
_Baseline cozume yaptiginiz iyilestirmeleri listeleyin._

### Phase 1: RAG

1. BM25 + embedding + hybrid retrieval ve AST-aware chunking eklendi
2. Retrieval metrikleri ve path-normalizasyonu uygulandi
3. Hybrid agirliklari recall/NDCG odakli ayarlandi

### Phase 2: Prompting

1. Structured output ve JSON extraction eklendi
2. CoT prompt sablonlari hazirlandi

### Phase 3: Fine-Tuning

1. Egitim verisi dogrulama ve raporlama
2. Stratified train/val split ve OpenAI formatlama
3. Fine-tuning API entegrasyonu

### Phase 4: Analytics

1. Failure analyzer ve raporlama
2. A/B deney çatısı, t-test ve Cohen's d

---

## 7. What I Would Do with More Time / Daha Fazla Zamanla Ne Yapardim

_If you had another 8-12 hours, what would you improve?_
_8-12 saat daha olsaydi neyi iyilestirirdiniz?_

```
- task_04 icin build hatasını çıkıp gerekli patch/postprocess düzeltmesinin yapılması
- Hybrid retrieval için daha iyi embedding modeli ve domain-adapted reranker denenmesi
- Precision odaklı ayarların (file-level rerank, path boosting) daha fazla optimize edilmesi
- Fine-tuning job optimize edilmesi ve learning rate optimizasyon yöntemlerin uygulanması
```

---

## 8. LLM / Model Information / LLM / Model Bilgisi

| Parameter / Parametre                                                   | Value / Deger                                                                       |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Model / Model                                                           | gpt-3.5-turbo-0125 (base), ft:gpt-3.5-turbo-0125:personal:ggf-case:DC4cOuBU (tuned) |
| Base URL / Temel URL                                                    | https://api.openai.com/v1                                                           |
| Temperature / Sicaklik                                                  | Default set)                                                                        |
| Max Tokens / Maks Token                                                 | Default                                                                             |
| Embedding Model (if used) / Embedding Modeli (kullanildiysa)            | all-MiniLM-L6-v2                                                                    |
| Vector DB (if used) / Vektor DB (kullanildiysa)                         | Qdrant (not used)                                                                   |
| Fine-tuned Model ID (if created) / Fine-tuned Model ID (olusturulduysa) | ft:gpt-3.5-turbo-0125:personal:ggf-case:DC4cOuBU                                    |

---

## 9. Environment / Ortam

| Component / Bilesen              | Version / Surum |
| -------------------------------- | --------------- |
| OS / Isletim Sistemi             | Linux Debian    |
| Node.js                          | 20.0            |
| Python                           | 3.11.2          |
| Docker (if used / kullanildiysa) | -               |
