"""
tfidf_matcher.py — Classical ML Match Score
============================================
WHAT THIS FILE DOES:
    Calculates how similar a resume is to a job description using
    TF-IDF vectorization and Cosine Similarity.

TECHNOLOGY USED:
    TF-IDF (Term Frequency - Inverse Document Frequency):
        A classical Machine Learning technique that converts text
        into numbers. Words that appear often in a document but rarely
        across all documents get a HIGH score (they are important).
        Common words like "the", "is", "and" get a LOW score.

        Example:
            "Python" appears 5 times in resume but rarely in other docs
            → HIGH TF-IDF score → important word

    Cosine Similarity:
        Measures the angle between two vectors (resume vector and
        job description vector). If they point in the same direction
        (angle = 0°) → similarity = 1.0 (100% match).
        If they point opposite ways → similarity = 0.0 (0% match).

    scikit-learn:
        Python library that provides TfidfVectorizer and
        cosine_similarity in just 2-3 lines of code.

WHY WE NEED THIS:
    This is the classical ML approach. Fast, reliable, interpretable.
    We compare its score against the Deep Learning score later.

HOW TO TEST:
    python tfidf_matcher.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import process_file, process_text


def tfidf_match_score(resume_text: str, job_text: str) -> dict:
    """
    DEFINITION: Calculates match percentage between a resume and
    job description using TF-IDF + Cosine Similarity.

    HOW IT WORKS (step by step):
        1. Clean both texts
        2. TfidfVectorizer converts them into number arrays (vectors)
        3. cosine_similarity measures how close these vectors are
        4. Result is a number from 0.0 to 1.0 (multiply by 100 = %)

    Args:
        resume_text : Raw or cleaned resume text
        job_text    : Raw or cleaned job description text

    Returns:
        dict with keys:
            score_percent   : match score as a percentage (0-100)
            matched_skills  : skills present in BOTH resume and job
            missing_skills  : skills in job but NOT in resume
            resume_skills   : all skills found in resume
            job_skills      : all skills found in job description
    """
    resume_data = process_text(resume_text)
    job_data    = process_text(job_text)

    resume_clean = resume_data["clean"]
    job_clean    = job_data["clean"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000,
    )

    tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score_percent = round(score * 100, 2)

    resume_skills = set(resume_data["skills"])
    job_skills    = set(job_data["skills"])
    matched       = sorted(resume_skills & job_skills)
    missing       = sorted(job_skills - resume_skills)

    return {
        "score_percent" : score_percent,
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skills" : sorted(resume_skills),
        "job_skills"    : sorted(job_skills),
    }


def skill_match_percent(matched: list, job_skills: list) -> float:
    """
    DEFINITION: Simple skill coverage percentage.

    How many of the required job skills does the resume cover?

    Example:
        Job needs: [python, sql, git, ml]  = 4 skills
        Resume has: [python, git]          = 2 matched
        Skill coverage = 2/4 * 100 = 50%
    """
    if not job_skills:
        return 0.0
    return round(len(matched) / len(job_skills) * 100, 2)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  STEP 2 TEST — tfidf_matcher.py")
    print("="*55)

    resume_data = process_file("data/resume.txt")
    job_data    = process_file("data/job.txt")

    result = tfidf_match_score(resume_data["raw"], job_data["raw"])

    print(f"\n  TF-IDF Match Score : {result['score_percent']}%")
    print(f"\n  Skills in Resume   : {result['resume_skills']}")
    print(f"\n  Skills in Job      : {result['job_skills']}")
    print(f"\n  Matched Skills ({len(result['matched_skills'])}):")
    for s in result["matched_skills"]:
        print(f"    [MATCH]   {s}")
    print(f"\n  Missing Skills ({len(result['missing_skills'])}):")
    for s in result["missing_skills"]:
        print(f"    [MISSING] {s}")

    coverage = skill_match_percent(result["matched_skills"], result["job_skills"])
    print(f"\n  Skill Coverage     : {coverage}%")
    print("\ntfidf_matcher.py — OK")