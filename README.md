# Cyberbullying Detection Using NLP (NB → BiLSTM → BERT)

This project builds a complete multi-class cyberbullying detection pipeline using **classical ML**, **deep learning**, and **transformers**.  
It analyzes a Kaggle dataset of **47,000+ tweets** and compares three models:

✅ Multinomial Naive Bayes  
✅ BiLSTM  
✅ BERT (Transformer – Final Model)

---

## 📌 Dataset
- **47,000+ tweets**  
- **6 classes**:  
  - age  
  - ethnicity  
  - gender  
  - religion  
  - other\_cyberbullying  
  - not\_cyberbullying  
- Highly imbalanced dataset → required careful evaluation

---

## 🔧 Workflow Overview

### 1️⃣ Data Cleaning
- Lowercasing  
- Removal of URLs, mentions, hashtags, emojis  
- Stopword removal  
- Lemmatization  
- Duplicate removal  
- Train–test split with stratification  

### 2️⃣ Feature Engineering
- TF–IDF (20k max features) for classical models  
- Tokenization + padding + attention masking for BERT  

---

## 🧪 Model 1 — Multinomial Naive Bayes (Baseline)
- Features: **TF–IDF**  
- **Accuracy:** 75.38%  
- **Macro F1:** 0.73  

Performs well on frequent classes (age, ethnicity, religion)  
Struggles on **not\_cyberbullying** due to dataset imbalance.

---

## 🧬 Model 2 — BiLSTM (Deep Learning)
- Embedding layer + Bidirectional LSTM + dropout regularization  
- Optimizer: Adam  
- Trained for multiple epochs  

**Performance:**
- **Accuracy:** 83%  
- **Macro F1:** 0.83  

Major improvement over NB, especially in minority classes.

---

## 🤖 Model 3 — BERT (Transformer – Final Model)
Model used: **bert-base-uncased**

- Tokenized with WordPiece tokenizer  
- Trained with AdamW + warmup schedule  
- Max sequence length: 128  

**Performance:**  
- **Accuracy:** 86.82%  
- **Macro F1:** 0.8646  

Best precision–recall balance across all 6 classes.

---

## 📊 Model Comparison

| Model                     | Accuracy | Macro F1 |
|--------------------------|----------|----------|
| Multinomial NB           | 75.38%   | 0.73     |
| BiLSTM                   | 83%      | 0.83     |
| BERT (Final)             | 86.82%   | 0.8646   |

---

## ✅ Key Takeaways
- BERT significantly improves minority-class F1 scores.  
- NB serves as a fast, interpretable baseline but struggles with imbalance.  
- BiLSTM hits a strong middle ground with better sequence understanding.  
- Transformers remain the most powerful for context-heavy text classification.

---

## 📁 Repository Structure
