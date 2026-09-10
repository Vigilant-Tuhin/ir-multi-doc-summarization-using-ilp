# Multi-Document Summarization using Integer Linear Programming (ILP)

An extractive summarizer that formulates sentence selection as an exact Integer Linear Program, following McDonald (2007) — *"A Study of Global Inference Algorithms in Multi-Document Summarization"*. Built for CS60092 (Information Retrieval), Assignment 3.

## Overview

Given a document, the summarizer scores each sentence for **relevance** (how informative it is about the whole document) and pairwise **redundancy** (how much it overlaps with other candidate sentences), then solves an ILP to pick the subset of sentences that maximizes total relevance minus redundancy, subject to a summary length budget of `K = 200` words.

### Scoring functions

- **Relevance:** `Rel(i) = 1/POS(t_i, D) + SIM(t_i, D)` — a sentence scores higher the earlier it appears in the document, and the more similar it is (cosine similarity over TF-IDF vectors) to the document as a whole.
- **Redundancy:** `Red(i, j) = SIM(t_i, t_j)` — cosine similarity between sentence pairs.

### ILP formulation

```
maximize   Σ α_i · Rel(i)  −  Σ α_ij · Red(i, j)
subject to Σ α_i · len(i) ≤ K          (summary word budget)
           α_ij ≤ α_i,  α_ij ≤ α_j     (pairwise variable only active if both sentences selected)
           α_i + α_j − α_ij ≤ 1
           α_i, α_ij ∈ {0, 1}
```

Solved exactly using `PuLP` with the CBC solver (20-second time limit per document).

## Files

| File | Purpose |
|---|---|
| `Assignment3_22MT30013_summarizer.py` | Task A — preprocesses the dataset, computes relevance/redundancy scores, solves the ILP, and writes one summary per line to `Assignment3_22MT30013_summary.txt`. |
| `Assignment3_22MT30013_evaluator.py` | Task B — computes ROUGE-1 and ROUGE-2 precision/recall/F1 for each generated summary against the reference (`highlights`), plus corpus-level averages. |
| `Assignment3_22MT30013_summary.txt` | Generated output — one extractive summary per document, in dataset order. |
| `README.md` | This file. |

## Dataset

[CNN/DailyMail dataset](https://drive.google.com/file/d/1UW-hA5xIeRbXYq541J1oyA6gThSFrnAA/view) (linked in the assignment spec) — a CSV with `article` and `highlights` (reference summary) columns. Rows whose reference summary exceeds 200 words are excluded from evaluation, per the assignment spec.

The dataset file is not committed to this repo (download it from the link above and pass its path on the command line).

## Setup

```bash
git clone https://github.com/Vigilant-Tuhin/ir-multi-doc-summarization-using-ilp.git
cd ir-multi-doc-summarization-using-ilp
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

## Usage

**Task A — generate summaries:**
```bash
python Assignment3_22MT30013_summarizer.py <path to data file>
```
Writes `Assignment3_22MT30013_summary.txt` to the working directory.

**Task B — evaluate against reference summaries:**
```bash
python Assignment3_22MT30013_evaluator.py <path to data file> <path to Assignment3_22MT30013_summary.txt>
```
Prints per-document and corpus-average ROUGE-1 / ROUGE-2 precision, recall, and F1 to the console.

## Dependencies

- `pandas`, `numpy`
- `nltk` (sentence tokenization — requires the `punkt` model)
- `scikit-learn` (TF-IDF vectorization, cosine similarity)
- `pulp` (ILP modeling, CBC solver)

See `requirements.txt` for the full list.

## Author

22MT30013
