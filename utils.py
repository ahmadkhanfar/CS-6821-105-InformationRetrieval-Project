# THIS FILE HAS THE METHODS FOR TOKEN, STOPWORD, STEEMING
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Tokenization functions
def tokenize_whitespace(text):
    """ Tokenizes based on whitespace """
    return text.split()

def tokenize_english(text):
    """ Tokenizes text using NLTK's word_tokenize for English """
    return word_tokenize(text)

# Stopword removal
def remove_stopwords(tokens):
    """ Removes stopwords from the tokenized list """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

# Stemming function
def apply_stemming(tokens):
    """ Applies stemming using NLTK's PorterStemmer to reduce words to their root form """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

