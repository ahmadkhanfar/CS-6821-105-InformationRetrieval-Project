import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from utils import tokenize_whitespace, tokenize_english, remove_stopwords, apply_stemming
import numpy as np
import nltk

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Directory for processed files
processed_dir = "Processed"
results_dir = "Results"

# Ensure processed and results directories exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Function to process and save CSV files
def process_and_save_csv(file_path, output_file, token_method, apply_stem, use_stopwords):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    if 'Text' not in df.columns:
        st.error(f"'Text' column not found in {file_path}")
        return None

    processed_data = []
    for text in df['Text']:
        tokens = tokenize_whitespace(text) if token_method == "whitespace" else tokenize_english(text)
        if use_stopwords:
            tokens = remove_stopwords(tokens)
        if apply_stem:
            tokens = apply_stemming(tokens)
        processed_data.append(" ".join(tokens))

    # Add processed text as a new column and save the result
    df['Processed_Text'] = processed_data
    processed_path = os.path.join(processed_dir, output_file)
    df.to_csv(processed_path, index=False)
    st.write(f"Processed file saved to {processed_path}")
    return processed_path

# Function to load or process files dynamically
def load_processed_file(file_name, token_method, apply_stem, use_stopwords):
    output_file = f"processed_{file_name}_{token_method}_{'stem' if apply_stem else 'no_stem'}_{'stopwords' if use_stopwords else 'no_stopwords'}.csv"
    processed_path = os.path.join(processed_dir, output_file)

    # Process and save file if not already processed
    if not os.path.exists(processed_path):
        st.write(f"Processing {file_name} with selected settings...")
        process_and_save_csv(file_name, output_file, token_method, apply_stem, use_stopwords)

    # Load processed file
    try:
        return pd.read_csv(processed_path)['Processed_Text'].tolist()
    except FileNotFoundError:
        st.error(f"Failed to create or load processed file: {processed_path}")
        return []

def compute_metrics(relevance_scores, k=10):
    # Precision, Recall, and F1-Score
    relevant_count = sum(relevance_scores)
    retrieved_relevant = sum(relevance_scores[:k])  # Relevant documents in top k
    total_retrieved = len(relevance_scores)
    total_relevant = sum(relevance_scores)  # Total relevant documents

    precision = retrieved_relevant / total_retrieved if total_retrieved > 0 else 0
    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Precision@k
    precision_at_k = retrieved_relevant / k if k > 0 else 0

    # NDCG
    dcg = sum([relevance_scores[i] / np.log2(i + 2) for i in range(min(k, len(relevance_scores)))])
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = sum([ideal_relevance_scores[i] / np.log2(i + 2) for i in range(min(k, len(ideal_relevance_scores)))])
    ndcg = dcg / idcg if idcg > 0 else 0

    return precision, recall, f1, precision_at_k, ndcg


# Save results to a CSV file
def save_results_to_csv(results, query, ranking_method):
    file_name = f"results_{ranking_method}.csv"
    results_path = os.path.join(results_dir, file_name)

    # Create a DataFrame and append the results
    df = pd.DataFrame([results], columns=["Query", "Precision", "Recall", "F1-Score", "P@10", "NDCG"])
    df["Query"] = query

    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)

    st.write(f"Results for query '{query}' saved to {results_path}")
def main():
    st.title("Dynamic CSV Tokenization and Query Evaluation with Metrics")

    # User options
    token_method = st.radio("Choose tokenization method:", ["Whitespace Tokenization", "NLTK Tokenization"])
    token_method = "whitespace" if token_method == "Whitespace Tokenization" else "nltk"
    apply_stem = st.radio("Apply Stemming?", ["Yes", "No"]) == "Yes"
    use_stopwords = st.radio("Remove Stopwords?", ["Yes", "No"]) == "Yes"
    ranking_method = st.selectbox("Select Ranking Method:", ["TF-IDF", "BM25"])
    query = st.text_input("Enter your query:", "Government")

    # Always process and save new files based on settings
    train_file = "bbc_train_sample.csv"
    test_file = "bbc_test_sample.csv"
    train_documents = load_processed_file(train_file, token_method, apply_stem, use_stopwords)
    test_documents = load_processed_file(test_file, token_method, apply_stem, use_stopwords)

    # If no processed data, alert the user
    if not train_documents or not test_documents:
        st.warning("No documents found after preprocessing. Check your input files or preprocessing settings.")
        return

    # Preprocess the query using the same settings
    query_tokens = tokenize_whitespace(query) if token_method == "whitespace" else tokenize_english(query)
    if use_stopwords:
        query_tokens = remove_stopwords(query_tokens)
    if apply_stem:
        query_tokens = apply_stemming(query_tokens)

    # Perform ranking based on the selected method
    if ranking_method == "TF-IDF":
        # Train TF-IDF model
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(train_documents)

        # Evaluate on test data
        test_tfidf_matrix = vectorizer.transform(test_documents)
        query_vector = vectorizer.transform([" ".join(query_tokens)])
        cosine_similarities = (test_tfidf_matrix * query_vector.T).toarray().flatten()

        # Sort results
        sorted_indices = cosine_similarities.argsort()[::-1]
        relevance_scores = [1 if idx < 10 else 0 for idx in sorted_indices]  # Simulate relevance: top 10 are relevant

    elif ranking_method == "BM25":
        # Train BM25 model
        tokenized_train_docs = [doc.split() for doc in train_documents]
        bm25 = BM25Okapi(tokenized_train_docs)

        # Evaluate on test data
        tokenized_test_docs = [doc.split() for doc in test_documents]
        bm25_scores = bm25.get_scores(query_tokens)
        sorted_indices = bm25_scores.argsort()[::-1]
        relevance_scores = [1 if idx < 10 else 0 for idx in sorted_indices]  # Simulate relevance: top 10 are relevant

    # Compute metrics
    precision, recall, f1, p_at_k, ndcg = compute_metrics(relevance_scores)

    # Save results
    save_results_to_csv([query, precision, recall, f1, p_at_k, ndcg], query, ranking_method)

    # Display results
    st.write(f"Results for query '{query}':")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")
    st.write(f"P@10: {p_at_k:.4f}")
    st.write(f"NDCG: {ndcg:.4f}")


if __name__ == "__main__":
    main()
