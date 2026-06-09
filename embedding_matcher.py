"""
embedding_matcher.py — Deep Learning Match Score
=================================================
WHAT THIS FILE DOES:
    Calculates match score using Sentence Embeddings — a Deep Learning
    approach that understands MEANING, not just keywords.

TECHNOLOGY USED:
    Sentence Transformers:
        A Deep Learning model based on BERT (Bidirectional Encoder
        Representations from Transformers). It reads an entire sentence
        and converts it into a 384-number vector that captures the
        MEANING of that sentence.

        Example of why embeddings are smarter than TF-IDF:
            TF-IDF sees "ML" and "Machine Learning" as DIFFERENT words.
            Embeddings understand they mean the SAME thing.

    all-MiniLM-L6-v2:
        The specific model we use. "Mini" means it is small and fast.
        Only 80 MB. Runs comfortably on your i3 + 8 GB laptop.
        Trained on 1 billion sentence pairs.

    Cosine Similarity:
        Same concept as in tfidf_matcher.py — measures angle between
        two meaning-vectors. Close angle = high semantic similarity.

DIFFERENCE FROM TF-IDF:
    TF-IDF matches KEYWORDS (exact words)
    Embeddings match MEANING (understands context and synonyms)

HOW TO TEST:
    python embedding_matcher.py
"""

from sentence_transformers import SentenceTransformer, util
from preprocess import process_file, process_text

print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def embedding_match_score(resume_text: str, job_text: str) -> dict:
    """
    DEFINITION: Calculates semantic similarity between resume and
    job description using sentence embeddings.

    HOW IT WORKS:
        1. Clean both texts using preprocess.py
        2. Model encodes each text into a 384-dimensional vector
           (a list of 384 numbers representing meaning)
        3. Cosine similarity compares the two meaning-vectors
        4. Score 0.0 to 1.0 → multiply by 100 for percentage

    Args:
        resume_text : Raw resume text
        job_text    : Raw job description text

    Returns:
        dict with keys:
            score_percent      : semantic similarity percentage (0-100)
            interpretation     : human-readable label (Excellent/Good/etc.)
            resume_word_count  : number of words in resume
            job_word_count     : number of words in job description
    """
    resume_clean = process_text(resume_text)["clean"]
    job_clean    = process_text(job_text)["clean"]

    resume_embedding = model.encode(resume_clean, convert_to_tensor=True)
    job_embedding    = model.encode(job_clean,    convert_to_tensor=True)

    similarity = util.cos_sim(resume_embedding, job_embedding).item()
    score_percent = round(similarity * 100, 2)

    if score_percent >= 80:
        interpretation = "Excellent match — very likely to get shortlisted"
    elif score_percent >= 60:
        interpretation = "Good match — strong candidate"
    elif score_percent >= 40:
        interpretation = "Moderate match — some gaps present"
    elif score_percent >= 20:
        interpretation = "Weak match — significant skills missing"
    else:
        interpretation = "Poor match — resume needs major updates"

    return {
        "score_percent"    : score_percent,
        "interpretation"   : interpretation,
        "resume_word_count": len(resume_clean.split()),
        "job_word_count"   : len(job_clean.split()),
    }


def get_section_scores(resume_text: str, job_sections: list) -> list:
    """
    DEFINITION: Scores a resume against individual sections
    of a job description separately.

    Useful when a job description has multiple sections like
    "Requirements", "Responsibilities", "Nice to have".

    Args:
        resume_text  : Raw resume text
        job_sections : List of job description section strings

    Returns:
        List of dicts, each with 'section_preview' and 'score_percent'
    """
    resume_clean = process_text(resume_text)["clean"]
    resume_emb   = model.encode(resume_clean, convert_to_tensor=True)

    results = []
    for section in job_sections:
        sec_clean = process_text(section)["clean"]
        sec_emb   = model.encode(sec_clean, convert_to_tensor=True)
        sim       = util.cos_sim(resume_emb, sec_emb).item()
        results.append({
            "section_preview": section[:60] + "...",
            "score_percent"  : round(sim * 100, 2),
        })
    return sorted(results, key=lambda x: x["score_percent"], reverse=True)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  STEP 3 TEST — embedding_matcher.py")
    print("="*55)

    resume = process_file("data/resume.txt")["raw"]
    job    = process_file("data/job.txt")["raw"]

    result = embedding_match_score(resume, job)

    print(f"\n  Embedding Score  : {result['score_percent']}%")
    print(f"  Interpretation   : {result['interpretation']}")
    print(f"  Resume words     : {result['resume_word_count']}")
    print(f"  Job desc words   : {result['job_word_count']}")
    print("\nembedding_matcher.py — OK")