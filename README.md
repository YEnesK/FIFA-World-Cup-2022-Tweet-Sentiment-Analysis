# FIFA World Cup 2022 Tweet Sentiment Analysis ⚽

A multi-approach sentiment analysis system that classifies FIFA World Cup 2022 tweets into **positive**, **neutral**, and **negative** categories using four different methods — from a simple neural network to fine-tuned transformer models.

## 📌 Overview

This project performs English sentiment analysis on tweets posted during the 2022 FIFA World Cup. Four approaches are compared to evaluate the trade-offs between model complexity and performance:

| # | Method | Accuracy | F1 (Macro) |
|---|--------|----------|------------|
| 1 | MiniLM Embeddings + MLP | 75.80% | 0.76 |
| 2 | **Fine-tuned RoBERTa (Classification)** | **88.13%** | **0.88** |
| 3 | Fine-tuned RoBERTa (Regression) | — | — |
| 4 | Regression → Classification (Optimal Thresholds) | 87.28% | 0.87 |

> The fine-tuned classification model achieved the best overall accuracy, while the regression model excelled at detecting neutral expressions.

## 📂 Project Structure

```
nlp/
├── dataset.ipynb          # Data loading, balancing, splitting & embedding generation
├── simple.ipynb           # Approach 1: MiniLM embeddings + MLP classifier
├── finetune.ipynb         # Approach 2: Fine-tuned RoBERTa classification
├── regression.ipynb       # Approach 3 & 4: RoBERTa regression + threshold conversion
├── fifa_world_cup_2022_tweets.csv   # Original dataset
├── combined_tweets.csv    # Processed & balanced dataset
├── tweet_embeddings.pkl   # Pre-computed MiniLM embeddings
├── tweet_sentiment_model.h5  # Saved Keras MLP model
├── Rapor_24501097.pdf     # Project report (Turkish)
└── README.md
```

## 📊 Dataset

- **Source:** FIFA World Cup 2022 Twitter data
- **Total tweets:** 22,524
- **Labels:** Positive, Neutral, Negative
- **Balancing:** 5,784 tweets per class (random sampling)
- **Split:** 70% train / 15% validation / 15% test (stratified)

## 🔬 Approaches

### 1. Simple Neural Network (MLP)
- Tweets embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
- 3-layer MLP with dropout (128 → 64 → 32 → 3)
- Trained with Adam optimizer and early stopping
- **Result:** 75.80% accuracy

### 2. Fine-tuned RoBERTa Classification
- Base model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Fine-tuned on project dataset using HuggingFace Trainer
- Early stopping on validation loss (patience=5)
- **Result:** 88.13% accuracy, F1 0.88

### 3. RoBERTa Regression
- Same base model with a custom regression head
- Predicts continuous sentiment intensity on [-1, 1] scale
  - -1 = most negative, 0 = neutral, +1 = most positive
- Regression metrics: Correlation 0.90, RMSE 0.36

### 4. Regression → Classification (Optimal Thresholds)
- Regression outputs converted to classes using optimized thresholds
  - Negative threshold: -0.305
  - Positive threshold: 0.715
- **Result:** 87.28% accuracy, F1 0.87
- Best model for detecting neutral tweets

## 🛠️ Technologies

- **Python**, **Jupyter Notebook**, **Google Colab**
- **Transformers** (HuggingFace) — RoBERTa fine-tuning
- **Sentence-Transformers** — MiniLM embeddings
- **TensorFlow / Keras** — Simple MLP model
- **PyTorch** — Custom regression head
- **scikit-learn** — Metrics, splitting, evaluation

## 🚀 Getting Started

### Prerequisites
```bash
pip install transformers sentence-transformers torch tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### Running the Notebooks
1. **`dataset.ipynb`** — Run first to prepare data and generate embeddings
2. **`simple.ipynb`** — MLP classifier using pre-computed embeddings
3. **`finetune.ipynb`** — Fine-tune RoBERTa for classification
4. **`regression.ipynb`** — Train regression model and convert to classification

> **Note:** Notebooks were developed on Google Colab with GPU (Tesla T4). Adjust paths if running locally.

## 📈 Key Findings

- The **fine-tuned RoBERTa classification** model delivers the best overall accuracy (88.13%) and excels at identifying positive and negative tweets.
- The **regression-to-classification** approach achieves competitive results (87.28%) while providing the additional benefit of continuous sentiment scores.
- The regression model with **optimal thresholds** is the best at detecting **neutral expressions**.
- Even a **simple MLP** with lightweight embeddings provides a reasonable baseline (75.80%) with minimal computational resources.

## 👤 Author

**Yusuf Enes Kurt**
- Student ID: 24501097
- Email: enes.kurt1@std.yildiz.edu.tr
- GitHub: [YEnesK](https://github.com/YEnesK)

Course Instructor: **Prof. Dr. Banu Diri**
Yıldız Technical University — Natural Language Processing (Doğal Dil İşlemeye Kavramsal Bir Bakış)
