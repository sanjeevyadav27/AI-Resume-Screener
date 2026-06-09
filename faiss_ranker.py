"""
faiss_ranker.py — FAISS Job Ranking
=====================================
WHAT THIS FILE DOES:
    When you have MULTIPLE job descriptions, this file ranks ALL of them
    against your single resume — from best match to worst match.
    This is the most impressive part of the project for your resume.

TECHNOLOGY USED:
    FAISS (Facebook AI Similarity Search):
        A library built by Facebook AI Research specifically for
        searching through large collections of vectors FAST.

        DEFINITION: FAISS builds an INDEX of all job description
        vectors, then searches that index to find which vectors are
        closest to the resume vector.

        Why FAISS instead of just cosine_similarity?
            cosine_similarity compares 1 resume to 1 job → fine for small data
            FAISS compares 1 resume to 1000+ jobs in milliseconds → scales up

        In this project, FAISS lets us say:
            "Here is my resume. Find the TOP 3 best-matching jobs
             from a list of 10 jobs, ranked by match score."

    sentence-transformers:
        Same model from embedding_matcher.py — converts each job
        description into a 384-number meaning-vector.

    numpy:
        Converts vectors to float32 format that FAISS requires.

HOW TO TEST:
    python faiss_ranker.py
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import process_text

print("Loading model for FAISS ranker...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.")


def build_faiss_index(job_texts: list) -> tuple:
    """
    DEFINITION: Builds a FAISS index from a list of job descriptions.

    Think of it like building a library catalogue:
        - Each job description = a book
        - Its embedding = the book's location code
        - FAISS index = the catalogue that lets you find similar books fast

    Args:
        job_texts : List of raw job description strings

    Returns:
        Tuple of (faiss_index, embeddings_array)
            faiss_index      : searchable index object
            embeddings_array : numpy array of all job embeddings
    """
    cleaned     = [process_text(t)["clean"] for t in job_texts]
    embeddings  = model.encode(cleaned, convert_to_numpy=True)
    embeddings  = embeddings.astype("float32")

    # Normalize so cosine similarity = dot product (FAISS uses dot product)
    faiss.normalize_L2(embeddings)

    dimension   = embeddings.shape[1]           # 384 for all-MiniLM-L6-v2
    index       = faiss.IndexFlatIP(dimension)  # IP = Inner Product = cosine similarity
    index.add(embeddings)

    return index, embeddings


def rank_jobs(resume_text: str, job_texts: list, job_titles: list, top_k: int = None) -> list:
    """
    DEFINITION: Ranks all job descriptions by how well they match
    a given resume, from highest to lowest match score.

    HOW IT WORKS:
        1. Encode resume text into a 384-number vector
        2. Normalize the vector
        3. Build FAISS index from all job descriptions
        4. FAISS searches index and returns top_k closest job vectors
        5. Return ranked list with scores and match details

    Args:
        resume_text : Raw resume text
        job_texts   : List of raw job description strings
        job_titles  : List of job title strings (for display)
        top_k       : How many top results to return (default = all)

    Returns:
        List of dicts sorted by score (highest first), each containing:
            rank, title, score_percent, interpretation
    """
    if top_k is None:
        top_k = len(job_texts)

    index, _ = build_faiss_index(job_texts)

    resume_clean = process_text(resume_text)["clean"]
    resume_emb   = model.encode([resume_clean], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(resume_emb)

    scores, indices = index.search(resume_emb, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        pct = round(float(score) * 100, 2)

        if pct >= 80:
            interp = "Excellent match"
        elif pct >= 60:
            interp = "Good match"
        elif pct >= 40:
            interp = "Moderate match"
        else:
            interp = "Weak match"

        results.append({
            "rank"         : rank,
            "title"        : job_titles[idx],
            "score_percent": pct,
            "interpretation": interp,
        })

    return results


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  STEP 4 TEST — faiss_ranker.py")
    print("="*55)

    with open("data/resume.txt", "r", encoding="utf-8") as f:
        resume = f.read()

    sample_jobs = [
        ("Junior Data Scientist",
         "Python machine learning scikit-learn pandas data analysis model training evaluation"),
        ("Frontend Developer",
         "HTML CSS JavaScript React UI UX web design responsive layout"),
        ("NLP Engineer",
         "Python NLP spaCy transformers BERT text classification deep learning HuggingFace"),
        ("Database Administrator",
         "MySQL PostgreSQL SQL database design indexing backup recovery Oracle"),
        ("ML Engineer",
         "Machine learning Python TensorFlow PyTorch model deployment FastAPI deep learning NLP"),
    ]

    titles = [j[0] for j in sample_jobs]
    descs  = [j[1] for j in sample_jobs]

    print("\n  Ranking jobs for your resume...\n")
    ranked = rank_jobs(resume, descs, titles)

    print(f"  {'Rank':<6} {'Job Title':<30} {'Score':>8}  Verdict")
    print(f"  {'-'*6} {'-'*30} {'-'*8}  {'-'*20}")
    for r in ranked:
        print(f"  #{r['rank']:<5} {r['title']:<30} {r['score_percent']:>7}%  {r['interpretation']}")

    print("\nfaiss_ranker.py — OK")