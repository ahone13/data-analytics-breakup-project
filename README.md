# 💔 The Anatomy of a Modern Breakup
### A Multi-Dataset Sentiment & Engagement Analytics Pipeline

**Course:** Data Analytics — DS26  
**Student:** Okah Ahone Ebwekoh  
**Supervisor:** Prof. Nor Azizah Hitam  
**University:** University of Europe for Applied Sciences  
**Submitted:** July 2026

---

## 📌 Project Overview

This project analyzes over 2,800 Reddit posts from r/breakups and r/relationship_advice to understand what drives community engagement in online relationship support communities. Using sentiment analysis, topic clustering, and supervised machine learning, the project tests the core hypothesis that structural factors — not emotional content — are the primary drivers of post visibility.

**Key finding:** Emotional sentiment has near-zero correlation with engagement. Comment count and post timing are the strongest predictors of whether a post goes viral — not how emotional the writing is. We call this the **Sentiment Paradox**.

---

## 📁 Project Structure

```
Final_Project_Submission/
│
├── Final_Report_v2.docx              → Full project report (8 sections)
│
├── data/
│   ├── raw/
│   │   ├── relationship_advice.csv   → Original r/relationship_advice dataset
│   │   └── reddit_breakup_dataset_cleaned.csv  → Original r/breakups dataset
│   └── processed/
│       └── merged_clean.csv          → Cleaned, merged dataset (output of notebook)
│
├── notebooks/
│   └── main_analysis.ipynb           → Full analytical pipeline
│
├── dashboards/
│   └── dashboard.py                  → Interactive Streamlit dashboard
│
├── diagrams/
│   ├── eda_overview.png              → EDA charts (score, length, engagement, hour)
│   ├── sentiment_analysis.png        → Sentiment distribution and boxplots
│   ├── wordclouds.png                → Word clouds (positive vs negative)
│   ├── correlation_heatmap.png       → Feature correlation matrix
│   ├── pmf_cdf.png                   → PMF and CDF of post score
│   ├── topic_distribution.png        → LDA topic clustering results
│   ├── confusion_matrices.png        → Baseline model confusion matrices
│   ├── confusion_matrices_improved.png → Improved model confusion matrices
│   ├── feature_importance.png        → RF importance & LR coefficients
│   ├── model_comparison.png          → Baseline model metrics comparison
│   └── improvement_comparison.png    → Before vs after recall improvement
│
├── documentation/
│   └── README.md                     → This file
│
└── references/
    └── citation_list.bib             → APA references
```

---

## 📊 Datasets

| Dataset | Source | Rows | Columns |
|---|---|---|---|
| relationship_advice.csv | Kaggle | 1,885 | 8 |
| reddit_breakup_dataset_cleaned.csv | Kaggle | 948 | 10 |
| **Combined (after cleaning)** | — | **2,825** | **12** |

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Text Cleaning | re (regex) |
| Sentiment Analysis | VADER (vaderSentiment) |
| Topic Modeling | Scikit-learn (LDA, CountVectorizer) |
| Machine Learning | Scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Dashboard | Streamlit |
| Environment | Jupyter Notebook (Anaconda) |

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn vaderSentiment wordcloud streamlit
```

### 2. Run the Jupyter Notebook
Open Anaconda Navigator → Launch Jupyter Notebook → Open `notebooks/main_analysis.ipynb` → Run All Cells

Make sure the datasets are in `data/raw/` before running.

### 3. Run the Streamlit Dashboard
```bash
cd "path/to/Final_Project_Submission"
streamlit run dashboards/dashboard.py
```
The dashboard will open automatically at `http://localhost:8501`

---

## 🔑 Key Results

| Approach | Best Recall | Best F1 |
|---|---|---|
| Baseline Models | 0.244 (RF) | 0.349 (RF) |
| Class Weighted | 0.944 (DT) | 0.630 (DT) |
| SMOTE | 0.861 (DT) | 0.634 (DT) |

**5 Topic Clusters identified:**
- Heartbreak & Recovery — 30.4%
- Long-term Strain — 21.3%
- Dating & Early Relationships — 17.9%
- Intimacy & Dynamics — 16.2%
- Social & Family Conflict — 14.4%

---

## 📚 References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

Shujon, S. (2025). Reddit break up stories dataset 2023–2025. Kaggle.

The Devastator. (2025). Unveiling relationship dynamics with Reddit. Kaggle.
