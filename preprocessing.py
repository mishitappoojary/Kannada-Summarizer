import re
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from indic_transliteration import sanscript
import pandas as pd
from indic_transliteration.sanscript import transliterate

# Load your CSV data
df = pd.read_csv('kannada_news_articles.csv')

# Kannada stopwords
stopwords_kannada = set([
    'ಈ', 'ಮತ್ತು', 'ಹಾಗೂ', 'ಅವರು', 'ಅವರ', 'ಬಗ್ಗೆ', 'ಎಂಬ', 'ಆದರೆ', 'ಅವರನ್ನು',
    'ತಮ್ಮ', 'ಒಂದು', 'ಎಂದರು', 'ಮೇಲೆ', 'ಹೇಳಿದರು', 'ಸೇರಿದಂತೆ', 'ಬಳಿಕ', 'ಆ',
    'ಯಾವುದೇ', 'ಅವರಿಗೆ', 'ನಡೆದ', 'ಕುರಿತು', 'ಇದು', 'ಕಳೆದ', 'ಇದೇ', 'ತಿಳಿಸಿದರು',
    'ಹೀಗಾಗಿ', 'ಕೂಡ', 'ತನ್ನ', 'ತಿಳಿಸಿದ್ದಾರೆ', 'ನಾನು', 'ಹೇಳಿದ್ದಾರೆ', 'ಈಗ', 'ಎಲ್ಲ',
    'ನನ್ನ', 'ನಮ್ಮ', 'ಈಗಾಗಲೇ', 'ಇದಕ್ಕೆ', 'ಹಲವು', 'ಇದೆ', 'ಮತ್ತೆ', 'ಮಾಡುವ', 'ನೀಡಿದರು',
    'ನಾವು', 'ನೀಡಿದ', 'ಇದರಿಂದ', 'ಅದು', 'ಇದನ್ನು', 'ನೀಡಿದ್ದಾರೆ', 'ಅದನ್ನು', 'ಇಲ್ಲಿ',
    'ಆಗ', 'ಬಂದಿದೆ', 'ಅದೇ', 'ಇರುವ', 'ಅಲ್ಲದೆ', 'ಕೆಲವು', 'ನೀಡಿದೆ', 'ಇದರ', 'ಇನ್ನು',
    'ನಡೆದಿದೆ'
])

def is_kannada_script(text):
    kannada_range = re.compile('[\u0C80-\u0CFF]')
    return bool(kannada_range.search(text))

# Preprocessing steps
def clean_text(text):
# Script validation: Keep only Kannada characters
    if not is_kannada_script(text):
        return ''  # Return an empty string if text is not in Kannada script

    # Remove characters outside Kannada script
    text = re.sub(r'[^\u0C80-\u0CFF\s]', '', text)

    # Tokenize the text
    tokens = regexp_tokenize(text, pattern=r'\s+', gaps=True)  # Using regexp_tokenize for better tokenization

    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords_kannada]

    # Reconstruct the cleaned text
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Apply preprocessing to the 'summary' and 'full_text' columns
df['preprocessed_summary'] = df['summary'].apply(clean_text)
df['preprocessed_full_text'] = df['full_text'].apply(clean_text)
df['preprocessed_title'] = df['title'].apply(clean_text)

# Remove rows with empty preprocessed summaries or full texts
df = df[df['preprocessed_summary'].str.strip().astype(bool) & df['preprocessed_full_text'].str.strip().astype(bool) & df['preprocessed_title'].str.strip().astype(bool)]
# Save the preprocessed data to a new CSV
df.to_csv('preprocessed_kannada_news_articles_py.csv', index=False)
print("Preprocessing completed and data saved to preprocessed_kannada_news_articles.csv")

def romanize_text(text):
    # Transliterate Kannada text to Romanized text
    return transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
    df = pd.read_csv('preprocessed_kannada_news_articles_py.csv')

print(df['summary'].apply(clean_text))
