import pandas as pd
import re
import numpy as np
import nltk
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
MAX_SUMMARY_WORDS = 200
LAMBDA = 1  # Weight parameter for redundancy control

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\?\!]', '', text)  # Remove other punctuation but keep ., ?, !
        return text.strip()
    return ""

# Preprocessing dataset without altering sentence boundaries
def preprocess_dataset(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    df['article'] = df['article'].apply(clean_text)
    df['highlights'] = df['highlights'].apply(clean_text)
    df['highlights'] = df['highlights'].apply(lambda x: "" if len(x.split()) > MAX_SUMMARY_WORDS else x)
    return df

# Tokenizing sentences using NLTK's Punkt
def punkt_sentence_tokenize(text):
    if not isinstance(text, str):
        return []
    return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]

# Calculate relevance scores using the formula: 1/POS(ti, D) + SIM(ti, D)
def calculate_relevance_scores(sentences, document):
    num_sentences = len(sentences)
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform([document] + sentences)
    doc_vector = tfidf_matrix[0]
    sent_vectors = tfidf_matrix[1:]

    sim_scores = cosine_similarity(sent_vectors, doc_vector).flatten()
    relevance_scores = [(1 / (i + 1)) + sim_scores[i] for i in range(num_sentences)]
    return relevance_scores, sent_vectors

# Convert csr_matrix to a tuple of tuples to make it hashable
def csr_matrix_to_tuple(csr):
    return tuple(map(tuple, csr.toarray()))

from functools import lru_cache

@lru_cache(maxsize=1000)
def calculate_similarity_matrix_cached(sentences_tuple):
    # Convert the tuple back to a csr_matrix
    vectorizer = TfidfVectorizer()
    sent_vectors = vectorizer.fit_transform(sentences_tuple)
    similarity_matrix = cosine_similarity(sent_vectors)
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix

def calculate_similarity_matrix(sent_vectors):
    sentences_tuple = tuple(sent_vectors)
    return calculate_similarity_matrix_cached(sentences_tuple)

# Calculate redundancy scores from similarity matrix
def calculate_redundancy_scores(similarity_matrix):
    num_sentences = similarity_matrix.shape[0]
    redundancy_scores = {(i, j): similarity_matrix[i, j] for i in range(num_sentences) for j in range(i + 1, num_sentences)}
    return redundancy_scores

# ILP Solver for sentence selection
def solve_ilp(sentences, relevance_scores, redundancy_scores, max_word_count, redundancy_penalty):
    num_sentences = len(sentences)
    prob = LpProblem("Sentence_Selection", LpMaximize)

    sentence_vars = [LpVariable(f"s_{i}", cat="Binary") for i in range(num_sentences)]
    pairwise_vars = {(i, j): LpVariable(f"p_{i}_{j}", cat="Binary") for (i, j) in redundancy_scores}

    prob += lpSum(sentence_vars[i] * relevance_scores[i] for i in range(num_sentences)) - \
            redundancy_penalty * lpSum(pairwise_vars[(i, j)] * redundancy_scores[(i, j)] for (i, j) in redundancy_scores)

    prob += lpSum(len(sentences[i].split()) * sentence_vars[i] for i in range(num_sentences)) <= max_word_count

    for (i, j) in redundancy_scores:
        prob += pairwise_vars[(i, j)] <= sentence_vars[i]
        prob += pairwise_vars[(i, j)] <= sentence_vars[j]
        prob += pairwise_vars[(i, j)] >= sentence_vars[i] + sentence_vars[j] - 1

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=20))

    selected_sentences = [sentences[i] for i in range(num_sentences) if sentence_vars[i].value() == 1]
    return selected_sentences if selected_sentences else []

# Generating summaries for the dataset
def generate_summaries(df):
    summaries = []
    for index, row in df.iterrows():
        document = row['article']
        sentences = punkt_sentence_tokenize(document)
        if len(sentences) < 3:
            summaries.append("Insufficient content for summary")
            continue

        relevance_scores, sent_vectors = calculate_relevance_scores(sentences, document)
        sentences_tuple = tuple(sentences)  # Convert sentences list to tuple
        similarity_matrix = calculate_similarity_matrix_cached(sentences_tuple)  # Use cached similarity matrix calculation
        redundancy_scores = calculate_redundancy_scores(similarity_matrix)

        summary_sentences = solve_ilp(sentences, relevance_scores, redundancy_scores, MAX_SUMMARY_WORDS, LAMBDA)
        summary = " ".join(summary_sentences)
        
        if len(summary.split()) <= MAX_SUMMARY_WORDS:
            summaries.append(summary)
        else:
            summaries.append("Summary exceeds word limit")
        
        print(f"Processed document {index + 1}/{len(df)}: Summary length = {len(summary.split())} words")
    return summaries

# Saving summaries to a file
def save_summary_file(summaries, file_name="Assignment3_22MT30013_summary.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        for summary in summaries:
            file.write(summary + "\n")

# Main function
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python Assignment3_22MT30013_summarizer.py <path_to_data_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = preprocess_dataset(file_path)
    summaries = generate_summaries(df)
    save_summary_file(summaries)
