import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from rank_bm25 import BM25Okapi  # For BM25
from utils import tokenize_whitespace, tokenize_english, remove_stopwords, apply_stemming

# Directory containing  CSV data files  (in our case we have bcc_test is inside the Data Direcotry)
data_dir = "Data"  # Ensure this path points to your CSV data folder

# Function to process CSV files and tokenize text
def process_and_tokenize_csv(data_dir, token_method="whitespace", apply_stem=True, use_stopwords=True):
    tokenized_data = []
    documents = []  # For holding original documents
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):  # Ensure we're working with CSV files
            file_path = os.path.join(data_dir, filename)
            st.write(f"Processing file: {filename}")  # Debugging output
            
            try:
                df = pd.read_csv(file_path)  # Load the CSV into a DataFrame
                # Check if 'Text' column exists in the file
                if 'Text' in df.columns:
                    for _, row in df.iterrows():
                        text = row['Text']  # Extract the 'Text' column
                        st.write(f"Original Text: {text}")  # Debugging output

                        # Apply the selected tokenization method
                        if token_method == "whitespace":
                            tokens = tokenize_whitespace(text)
                        else:
                            tokens = tokenize_english(text)

                        # Optionally apply stopword removal
                        if use_stopwords:
                            tokens = remove_stopwords(tokens)

                        # Apply stemming or non-stemming based on the user's choice
                        if apply_stem:
                            tokens_before_stem = tokens.copy()
                            tokens = apply_stemming(tokens)
                            st.write(f"Tokens before stemming: {tokens_before_stem}")
                            st.write(f"Tokens after stemming: {tokens}")
                        else:
                            st.write(f"Tokens without stemming: {tokens}")

                        # Append tokenized data with filename and document reference
                        tokenized_data.append({"docno": filename, "tokens": tokens})
                        documents.append(text)  # Storing original documents for ranking
                else:
                    st.write(f"Warning: 'Text' column not found in {filename}. Skipping file.")
            
            except Exception as e:
                st.write(f"Error processing {filename}: {e}")

    return tokenized_data, documents

# Function to compute TF (Term Frequency)
def compute_tf(documents):
    # Term Frequency calculation
    tf = []
    for doc in documents:
        words = doc.split()  # Split by whitespace
        word_count = len(words)
        term_freq = {word: words.count(word) / word_count for word in set(words)}
        tf.append(term_freq)
    return tf

# Function to compute TF-IDF
def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix

# Function to compute BM25
def compute_bm25(tokenized_data):
    bm25 = BM25Okapi([doc['tokens'] for doc in tokenized_data])
    return bm25

# Function to compute Precision, Recall, F1-Score
def compute_metrics(relevant_docs, retrieved_docs):
    precision = precision_score(relevant_docs, retrieved_docs, average='binary', zero_division=0)
    recall = recall_score(relevant_docs, retrieved_docs, average='binary', zero_division=0)
    f1 = f1_score(relevant_docs, retrieved_docs, average='binary', zero_division=0)
    return precision, recall, f1

# Function to calculate DCG and NDCG
def compute_dcg(relevance_scores, k=10):
    dcg = sum([relevance_scores[i] / np.log2(i + 2) for i in range(min(k, len(relevance_scores)))] )
    return dcg

def compute_ndcg(relevance_scores, k=10):
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = compute_dcg(ideal_relevance_scores, k)
    dcg = compute_dcg(relevance_scores, k)
    return dcg / idcg if idcg > 0 else 0

# Function to display the results in Streamlit
def display_results(metrics, ranking_results, ranking_method):
    st.write(f"Ranking Method: {ranking_method}")
    st.write(f"Precision: {metrics[0]:.4f}")
    st.write(f"Recall: {metrics[1]:.4f}")
    st.write(f"F1-Score: {metrics[2]:.4f}")
    
    st.write("Ranking Results (Top 10 documents):")
    for i, doc in enumerate(ranking_results[:10]):
        st.write(f"Rank {i + 1}: Document {doc}")

# Function to save the output to an Excel file
import pandas as pd
import streamlit as st
from openpyxl import Workbook  # Use openpyxl for Excel file handling

import pandas as pd
import streamlit as st
import os

import pandas as pd
import streamlit as st
import os

def save_to_excel(results, token_method, apply_stem, use_stopwords):
    # Prepare data for saving
    df = pd.DataFrame(results)

    # Add extra columns for the settings
    df['Tokenization Method'] = token_method
    df['Stemming'] = 'Yes' if apply_stem else 'No'
    df['Stopwords Removal'] = 'Yes' if use_stopwords else 'No'

    # Define the file name for the results
    file_name = "ranking_results.xlsx"

    try:
        # Check if the file already exists
        if os.path.exists(file_name):
            # If file exists, append the results to the existing file as a new row
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, sheet_name="Results", header=False, startrow=len(pd.read_excel(file_name, sheet_name="Results"))+1)
                st.write(f"New results added to {file_name}")
        else:
            # If the file doesn't exist, create a new file and write the results
            with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Results")
                st.write(f"Results saved to {file_name}")
    except Exception as e:
        st.write(f"Error saving to Excel: {e}")

    # Prepare data for saving
    df = pd.DataFrame(results)

    # Add extra columns for the settings
    df['Tokenization Method'] = token_method
    df['Stemming'] = 'Yes' if apply_stem else 'No'
    df['Stopwords Removal'] = 'Yes' if use_stopwords else 'No'

    # Define the file name for the results
    file_name = "ranking_results.xlsx"
    
    try:
        # Check if the file already exists
        if os.path.exists(file_name):
            # If file exists, append the results to the existing sheet
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, sheet_name="Results", header=False)
                st.write(f"Results appended to {file_name}")
        else:
            # If the file doesn't exist, create a new file and write the results
            with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Results")
                st.write(f"Results saved to {file_name}")
    except Exception as e:
        st.write(f"Error saving to Excel: {e}")

# Main Streamlit application
def main():
    st.title("CSV Tokenization, Ranking, and Evaluation")
      # Option to choose ranking method
    ranking_method = st.selectbox("Select Ranking Method:", ["TF", "TF-IDF", "BM25"])
    # Ask the user to select the tokenization method
    token_method = st.radio("Choose a tokenization method:", ["Whitespace Tokenization", "NLTK Tokenization"])
    token_method = "whitespace" if token_method == "Whitespace Tokenization" else "nltk"

    # Ask for stemming or non-stemming
    apply_stem = st.radio("Choose processing method:", ["Stemming", "Non-Stemming"]) == "Stemming"

    # Ask if stopwords should be removed
    use_stopwords = st.radio("Remove Stopwords?", ["Yes", "No"]) == "Yes"

    # Ask for a query
    query = st.text_input("Enter your query:", "Government")  # Default to "Government" query

    # Process CSV files and tokenize
    tokenized_data, documents = process_and_tokenize_csv(data_dir, token_method, apply_stem, use_stopwords)

  

    # Perform query-based ranking and save results
    ranking_results = []
    if ranking_method == "TF":
        tf = compute_tf(documents)
        query_tokens = query.split()
        query_tf = {word: query_tokens.count(word) / len(query_tokens) for word in set(query_tokens)}
        relevance_scores = [sum([query_tf.get(word, 0) * tf_doc.get(word, 0) for word in query_tokens]) for tf_doc in tf]
        sorted_docs = np.argsort(relevance_scores)[::-1]
        relevant_docs = [1 if query.lower() in doc.lower() else 0 for doc in documents]
        retrieved_docs = [1 if relevance_scores[i] > 0 else 0 for i in sorted_docs]
        precision, recall, f1 = compute_metrics(relevant_docs, retrieved_docs)
        display_results((precision, recall, f1), sorted_docs, "TF")
        results = {"Method": "TF", "Precision": precision, "Recall": recall, "F1-Score": f1, "Ranking Results": sorted_docs.tolist()}
        save_to_excel([results], token_method, apply_stem, use_stopwords)

    elif ranking_method == "TF-IDF":
        tfidf_matrix = compute_tfidf(documents)
        query_vector = TfidfVectorizer().fit(documents).transform([query])
        cosine_similarities = (tfidf_matrix * query_vector.T).toarray()
        relevant_docs = [1 if query.lower() in doc.lower() else 0 for doc in documents]
        retrieved_docs = [1 if sim > 0 else 0 for sim in cosine_similarities.flatten()]
        precision, recall, f1 = compute_metrics(relevant_docs, retrieved_docs)
        display_results((precision, recall, f1), cosine_similarities.flatten(), "TF-IDF")
        results = {"Method": "TF-IDF", "Precision": precision, "Recall": recall, "F1-Score": f1, "Ranking Results": cosine_similarities.flatten().tolist()}
        save_to_excel([results], token_method, apply_stem, use_stopwords)

    elif ranking_method == "BM25":
        query_tokens = tokenize_english(query) if token_method == "nltk" else tokenize_whitespace(query)
        bm25 = compute_bm25(tokenized_data)
        query_scores = bm25.get_scores(query_tokens)
        sorted_docs = np.argsort(query_scores)[::-1]
        relevant_docs = [1 if query.lower() in doc.lower() else 0 for doc in documents]
        retrieved_docs = [1 if query_scores[i] > 0 else 0 for i in sorted_docs]
        precision, recall, f1 = compute_metrics(relevant_docs, retrieved_docs)
        display_results((precision, recall, f1), sorted_docs, "BM25")
        results = {"Method": "BM25", "Precision": precision, "Recall": recall, "F1-Score": f1, "Ranking Results": sorted_docs.tolist()}
        save_to_excel([results], token_method, apply_stem, use_stopwords)


if __name__ == "__main__":
    main()
