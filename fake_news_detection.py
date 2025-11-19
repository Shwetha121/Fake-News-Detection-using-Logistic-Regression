"""
Fake News Detection using Logistic Regression
==============================================
This script implements a machine learning model to detect fake news articles
using Logistic Regression with TF-IDF vectorization.
"""

import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download required NLTK data
print("Downloading NLTK stopwords...")
nltk.download('stopwords')

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Loading Dataset")
print("="*70)

# Load the dataset
# Update the path to your dataset location
news_df = pd.read_csv("train.csv")

print(f"Dataset loaded successfully!")
print(f"Shape: {news_df.shape}")
print(f"\nFirst few rows:")
print(news_df.head())

# ============================================================================
# STEP 2: Data Preprocessing
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Data Preprocessing")
print("="*70)

# Check for missing values
print("\nMissing values before handling:")
print(news_df.isnull().sum())

# Fill missing values with empty string
news_df = news_df.fillna(' ')

print("\nMissing values after handling:")
print(news_df.isnull().sum())

# Combine author and title into content column
news_df['content'] = news_df['author'] + ' ' + news_df['title']

print("\nNew 'content' column created by combining 'author' and 'title'")
print(news_df[['author', 'title', 'content']].head())

# ============================================================================
# STEP 3: Text Preprocessing - Stemming
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Text Preprocessing - Stemming")
print("="*70)

# Initialize Porter Stemmer
ps = PorterStemmer()

def stemming(content):
    """
    Preprocess text by:
    1. Removing non-alphabetic characters
    2. Converting to lowercase
    3. Tokenizing
    4. Removing stopwords
    5. Stemming words
    
    Args:
        content (str): Input text
    
    Returns:
        str: Preprocessed text
    """
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    
    # Split into words
    stemmed_content = stemmed_content.split()
    
    # Remove stopwords and apply stemming
    stemmed_content = [ps.stem(word) for word in stemmed_content 
                       if word not in stopwords.words('english')]
    
    # Join words back into string
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content

# Apply stemming to content
print("Applying stemming to content... This may take a moment.")
news_df['content'] = news_df['content'].apply(stemming)

print("Stemming completed!")
print("\nExample of preprocessed content:")
print(news_df['content'].head())

# ============================================================================
# STEP 4: Separate Features and Labels
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Separating Features and Labels")
print("="*70)

# Separate the data (X) and labels (y)
X = news_df['content'].values
y = news_df['label'].values

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"\nLabel distribution:")
print(f"Real news (0): {sum(y == 0)}")
print(f"Fake news (1): {sum(y == 1)}")

# ============================================================================
# STEP 5: Convert Text to Numerical Data using TF-IDF
# ============================================================================
print("\n" + "="*70)
print("STEP 5: TF-IDF Vectorization")
print("="*70)

# Initialize TF-IDF Vectorizer
vector = TfidfVectorizer()

# Fit and transform the data
vector.fit(X)
X = vector.transform(X)

print(f"TF-IDF vectorization completed!")
print(f"Feature matrix shape: {X.shape}")
print(f"Number of unique words (features): {X.shape[1]}")

# ============================================================================
# STEP 6: Split Dataset into Training and Testing Sets
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Splitting Dataset")
print("="*70)

# Split the data (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ============================================================================
# STEP 7: Train the Logistic Regression Model
# ============================================================================
print("\n" + "="*70)
print("STEP 7: Training Logistic Regression Model")
print("="*70)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
print("Training model... Please wait.")
model.fit(X_train, Y_train)
print("Model training completed!")

# ============================================================================
# STEP 8: Evaluate Model Performance
# ============================================================================
print("\n" + "="*70)
print("STEP 8: Model Evaluation")
print("="*70)

# Predictions on training set
train_y_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_y_pred)
print(f"\n📊 Training Set Accuracy: {train_accuracy*100:.2f}%")

# Predictions on testing set
testing_y_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, testing_y_pred)
print(f"📊 Testing Set Accuracy: {test_accuracy*100:.2f}%")

# Detailed classification report
print("\n" + "-"*70)
print("Detailed Classification Report (Test Set):")
print("-"*70)
print(classification_report(Y_test, testing_y_pred, 
                          target_names=['Real News', 'Fake News']))

# Confusion Matrix
print("\n" + "-"*70)
print("Confusion Matrix (Test Set):")
print("-"*70)
cm = confusion_matrix(Y_test, testing_y_pred)
print(cm)
print("\nInterpretation:")
print(f"True Negatives (Real news correctly identified): {cm[0][0]}")
print(f"False Positives (Real news incorrectly identified as fake): {cm[0][1]}")
print(f"False Negatives (Fake news incorrectly identified as real): {cm[1][0]}")
print(f"True Positives (Fake news correctly identified): {cm[1][1]}")

# ============================================================================
# STEP 9: Test Detection System with Sample
# ============================================================================
print("\n" + "="*70)
print("STEP 9: Testing Detection System")
print("="*70)

# Test with a sample from test set
test_index = 10
input_data = X_test[test_index]
prediction = model.predict(input_data)

print(f"\nTest Sample #{test_index}:")
print(f"Original content: {news_df['content'].iloc[test_index][:100]}...")
print(f"\nActual Label: {'Real News' if Y_test[test_index] == 0 else 'Fake News'}")
print(f"Predicted Label: {'Real News' if prediction[0] == 0 else 'Fake News'}")

if prediction[0] == 0:
    print("\n✅ The News Is Real")
else:
    print("\n❌ The News is Fake")

# ============================================================================
# STEP 10: Interactive Prediction Function
# ============================================================================
print("\n" + "="*70)
print("STEP 10: Interactive Prediction Function")
print("="*70)

def predict_news(text):
    """
    Predict whether a news article is real or fake.
    
    Args:
        text (str): News article text
    
    Returns:
        str: Prediction result
    """
    # Preprocess the text
    processed_text = stemming(text)
    
    # Vectorize the text
    text_vector = vector.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    
    result = "Real News" if prediction[0] == 0 else "Fake News"
    confidence = max(probability[0]) * 100
    
    return result, confidence

# Example usage
print("\nExample Prediction:")
sample_text = "Breaking: Scientists discover amazing new cure for everything!"
result, confidence = predict_news(sample_text)
print(f"Text: '{sample_text}'")
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2f}%")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Model: Logistic Regression
Training Accuracy: {train_accuracy*100:.2f}%
Testing Accuracy: {test_accuracy*100:.2f}%

The model has been successfully trained and tested!
You can now use the predict_news() function to classify new articles.
""")

print("="*70)
print("Script execution completed successfully!")
print("="*70)