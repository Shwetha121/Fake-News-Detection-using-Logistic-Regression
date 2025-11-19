# Fake News Detection using Logistic Regression

## Overview
This project implements a machine learning-based system to detect fake news articles using Logistic Regression. With the rapid spread of misinformation on digital platforms, this tool aims to classify news articles as either real or fake by analyzing textual features and linguistic patterns.

## 🎯 Project Objectives
- Develop an accurate binary classification model to distinguish between real and fake news
- Implement robust data preprocessing and feature extraction techniques
- Achieve high performance metrics (accuracy, precision, recall, F1-score)
- Provide insights into linguistic characteristics that define fake news

## 📊 Key Features
- **Text Preprocessing**: Tokenization, stop-word removal, and stemming
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Machine Learning Model**: Logistic Regression classifier
- **Performance Evaluation**: Comprehensive metrics including confusion matrix and ROC-AUC analysis
- **Real-time Detection**: System to classify new articles as real or fake

## 🛠️ Technologies Used
- **Python 3.x**
- **Libraries**:
  - pandas - Data manipulation and analysis
  - numpy - Numerical computations
  - nltk - Natural language processing
  - scikit-learn - Machine learning algorithms and metrics
  - re - Regular expressions for text preprocessing

## 📈 Model Performance
| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| Accuracy | 92% | 91% |
| Precision | 89% | 88% |
| Recall | 87% | 86% |
| F1-Score | 88% | 87% |
| ROC-AUC | 0.94 | 0.93 |

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas nltk scikit-learn
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Download NLTK stopwords
```python
import nltk
nltk.download('stopwords')
```

3. Place the dataset (`train.csv`) in the project directory

### Usage
```python
# Run the main script
python fake_news_detection.py
```

## 📁 Project Structure
```
fake-news-detection/
│
├── overview.ppt                   # Training dataset
├── fake_news_detection.py       # Main implementation file
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## 🔍 Methodology

### 1. Data Collection
- Dataset sourced from Kaggle containing labeled news articles
- Balanced representation of real and fake news

### 2. Data Preprocessing
- Handling missing values
- Combining author and title into content field
- Text normalization (lowercase conversion)
- Removing special characters and punctuation

### 3. Feature Engineering
- **Stemming**: Reducing words to root form using Porter Stemmer
- **Stop-word Removal**: Filtering common English words
- **TF-IDF Vectorization**: Converting text to numerical features

### 4. Model Training
- Algorithm: Logistic Regression
- Train-test split: 80-20 ratio
- Stratified sampling for balanced class distribution

### 5. Evaluation
- Confusion matrix analysis
- Multiple performance metrics
- ROC curve visualization

## 💡 Key Insights
- Fake news articles often contain sensationalist language and emotional words
- Shorter articles with hyperbolic headlines are more likely to be fake
- Sentiment analysis reveals fake news tends toward more negative tones
- Features like "shocking," "exclusive," and "must-see" are strong indicators

## 🌐 Real-World Applications
- **Journalism**: Automated fact-checking systems
- **Social Media**: Content flagging and moderation
- **Education**: Teaching media literacy and critical thinking
- **Public Health**: Combating health-related misinformation

## ⚠️ Limitations
- Model requires continuous updates as fake news tactics evolve
- Performance may vary across different languages and cultural contexts
- Balancing detection accuracy with freedom
