import pandas as pd
import re
from collections import Counter
from nltk.util import ngrams
import sys

# Function to clean text (same as used in Task A for consistency)
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip().lower()  # Lowercase for consistent comparison
    return ""

# Load dataset and generated summaries
def load_data(data_file_path, summary_file_path):
    df = pd.read_csv(data_file_path)
    with open(summary_file_path, 'r', encoding="utf-8") as f:
        generated_summaries = [line.strip() for line in f]
    
    # Ensure both lists have the same length
    print(len(df))
    print(len(generated_summaries))
    assert len(df) == len(generated_summaries), "Mismatch in dataset and summary lengths."
    
    # Clean and prepare highlights for evaluation
    df['highlights'] = df['highlights'].apply(clean_text)
    
    return df['highlights'].tolist(), generated_summaries

# Function to compute ROUGE-1 score
def compute_rouge_1(reference, generated):
    ref_words = reference.split()
    gen_words = generated.split()
    
    # Count overlaps using Counter for word frequency
    ref_count = Counter(ref_words)
    gen_count = Counter(gen_words)
    
    overlap = sum(min(ref_count[word], gen_count[word]) for word in ref_count)
    
    precision = overlap / len(gen_words) if len(gen_words) > 0 else 0
    recall = overlap / len(ref_words) if len(ref_words) > 0 else 0
    
    # F1 score as the harmonic mean of precision and recall
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# Function to compute ROUGE-2 score
def compute_rouge_2(reference, generated):
    ref_bigrams = list(ngrams(reference.split(), 2))
    gen_bigrams = list(ngrams(generated.split(), 2))
    
    ref_count = Counter(ref_bigrams)
    gen_count = Counter(gen_bigrams)
    
    overlap = sum(min(ref_count[bigram], gen_count[bigram]) for bigram in ref_count)
    
    precision = overlap / len(gen_bigrams) if len(gen_bigrams) > 0 else 0
    recall = overlap / len(ref_bigrams) if len(ref_bigrams) > 0 else 0
    
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# Evaluate ROUGE scores for all documents
def evaluate_rouge_scores(reference_summaries, generated_summaries):
    rouge1_scores = []
    rouge2_scores = []

    # Initialize accumulators for average calculation
    total_rouge1_precision = total_rouge1_recall = total_rouge1_f1 = 0
    total_rouge2_precision = total_rouge2_recall = total_rouge2_f1 = 0

    for i in range(len(reference_summaries)):
        reference = reference_summaries[i]
        generated = generated_summaries[i]
        
        # Compute ROUGE-1
        rouge1_precision, rouge1_recall, rouge1_f1 = compute_rouge_1(reference, generated)
        rouge1_scores.append((rouge1_precision, rouge1_recall, rouge1_f1))
        
        # Compute ROUGE-2
        rouge2_precision, rouge2_recall, rouge2_f1 = compute_rouge_2(reference, generated)
        rouge2_scores.append((rouge2_precision, rouge2_recall, rouge2_f1))
        
        # Accumulate scores for averaging
        total_rouge1_precision += rouge1_precision
        total_rouge1_recall += rouge1_recall
        total_rouge1_f1 += rouge1_f1

        total_rouge2_precision += rouge2_precision
        total_rouge2_recall += rouge2_recall
        total_rouge2_f1 += rouge2_f1
        
        
        # Print scores for the current document
        print(f"Document {i+1}:")
        print(f"ROUGE-1: Precision={rouge1_precision:.4f}, Recall={rouge1_recall:.4f}, F1-Score={rouge1_f1:.4f}")
        print(f"ROUGE-2: Precision={rouge2_precision:.4f}, Recall={rouge2_recall:.4f}, F1-Score={rouge2_f1:.4f}\n")

    # Calculate average ROUGE-1 and ROUGE-2 scores
    num_docs = len(reference_summaries)
    avg_rouge1_precision = total_rouge1_precision / num_docs
    avg_rouge1_recall = total_rouge1_recall / num_docs
    avg_rouge1_f1 = total_rouge1_f1 / num_docs

    avg_rouge2_precision = total_rouge2_precision / num_docs
    avg_rouge2_recall = total_rouge2_recall / num_docs
    avg_rouge2_f1 = total_rouge2_f1 / num_docs

    # Print average scores
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1: Precision={avg_rouge1_precision:.4f}, Recall={avg_rouge1_recall:.4f}, F1-Score={avg_rouge1_f1:.4f}")
    print(f"ROUGE-2: Precision={avg_rouge2_precision:.4f}, Recall={avg_rouge2_recall:.4f}, F1-Score={avg_rouge2_f1:.4f}")


    return rouge1_scores, rouge2_scores

# Main function
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Assignment3_22MT30013_evaluator.py <path_to_data_file> <path_to_summary_file>")
        sys.exit(1)
    
    data_file_path = sys.argv[1]
    summary_file_path = sys.argv[2]
    
    # Load data
    reference_summaries, generated_summaries = load_data(data_file_path, summary_file_path)
    
    # Evaluate and print ROUGE scores
    evaluate_rouge_scores(reference_summaries, generated_summaries)
