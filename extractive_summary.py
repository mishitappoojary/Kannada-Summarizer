import pandas as pd
import numpy as np
import nltk
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle

# Load your preprocessed dataset
df = pd.read_csv('preprocessed_kannada_news_articles.csv')

# Extract sentences or paragraphs from your dataset
texts = df['preprocessed_full_text'].tolist()

# Define your stopwords list
stop_words = set([
    'ಈ', 'ಮತ್ತು', 'ಹಾಗೂ', 'ಅವರು', 'ಅವರ', 'ಬಗ್ಗೆ', 'ಎಂಬ', 'ಆದರೆ', 'ಅವರನ್ನು',
    'ತಮ್ಮ', 'ಒಂದು', 'ಎಂದರು', 'ಮೇಲೆ', 'ಹೇಳಿದರು', 'ಸೇರಿದಂತೆ', 'ಬಳಿಕ', 'ಆ',
    'ಯಾವುದೇ', 'ಅವರಿಗೆ', 'ನಡೆದ', 'ಕುರಿತು', 'ಇದು', 'ಕಳೆದ', 'ಇದೇ', 'ತಿಳಿಸಿದರು',
    'ಹೀಗಾಗಿ', 'ಕೂಡ', 'ತನ್ನ', 'ತಿಳಿಸಿದ್ದಾರೆ', 'ನಾನು', 'ಹೇಳಿದ್ದಾರೆ', 'ಈಗ', 'ಎಲ್ಲ',
    'ನನ್ನ', 'ನಮ್ಮ', 'ಈಗಾಗಲೇ', 'ಇದಕ್ಕೆ', 'ಹಲವು', 'ಇದೆ', 'ಮತ್ತೆ', 'ಮಾಡುವ', 'ನೀಡಿದರು',
    'ನಾವು', 'ನೀಡಿದ', 'ಇದರಿಂದ', 'ಅದು', 'ಇದನ್ನು', 'ನೀಡಿದ್ದಾರೆ', 'ಅದನ್ನು', 'ಇಲ್ಲಿ',
    'ಆಗ', 'ಬಂದಿದೆ', 'ಅದೇ', 'ಇರುವ', 'ಅಲ್ಲದೆ', 'ಕೆಲವು', 'ನೀಡಿದೆ', 'ಇದರ', 'ಇನ್ನು',
    'ನಡೆದಿದೆ'
])

# Preprocess the new text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word not in stop_words]
        processed_sentences.append(' '.join(filtered_words))
    return processed_sentences

# TF-IDF based scoring
def create_tfidf_matrix(texts, new_text_sentences):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    tfidf_matrix_new = vectorizer.transform(new_text_sentences)
    return tfidf_matrix_new

# Save the TF-IDF vectorizer
def save_vectorizer(vectorizer, filename='vectorizer.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(vectorizer, f)

# Load the TF-IDF vectorizer
def load_vectorizer(filename='vectorizer.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def preprocess_and_save_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    save_vectorizer(vectorizer)
    
preprocess_and_save_vectorizer(texts)

def extractive_summary(text, num_sentences=3):
    vectorizer = load_vectorizer()
    new_sentences = preprocess_text(text)
    tfidf_matrix_new = create_tfidf_matrix(texts, new_sentences)
    
    def score_sentences(tfidf_matrix):
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        return sentence_scores
    
    def sentence_length_score(sentences):
        return [len(sentence.split()) for sentence in sentences]
    
    def combine_scores(tfidf_scores, length_scores, weight_tfidf=0.7, weight_length=0.3):
        combined_scores = weight_tfidf * tfidf_scores + weight_length * np.array(length_scores)
        return combined_scores
    
    def generate_summary(sentences, sentence_scores, num_sentences=3):
        ranked_sentences = [sent for sent, score in sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)]
        summary = ' '.join(ranked_sentences[:num_sentences])
        return summary
    
    sentence_scores_new = score_sentences(tfidf_matrix_new)
    length_scores = sentence_length_score(new_sentences)
    combined_scores = combine_scores(sentence_scores_new, length_scores)
    summary = generate_summary(new_sentences, combined_scores, num_sentences=num_sentences)
    
    return summary

