"""
preprocess.py — Text Cleaning + Skill Extraction
=================================================
Supports BOTH:
✔ File input (CLI)
✔ Direct text input (Streamlit UI)
"""

import re
import spacy

print("Loading spaCy language model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy ready.")

TECH_SKILLS = {
    "python", "java", "c++", "javascript", "sql", "r",
    "machine learning", "deep learning", "nlp",
    "natural language processing", "scikit-learn",
    "tensorflow", "pytorch", "keras", "pandas", "numpy",
    "matplotlib", "seaborn", "fastapi", "flask", "django",
    "rest api", "faiss", "transformers", "huggingface",
    "bert", "gpt", "sentence-transformers", "spacy", "nltk",
    "git", "github", "docker", "aws", "azure",
    "mysql", "postgresql", "mongodb", "sqlite",
    "excel", "power bi", "tableau", "linux", "bash",
    "html", "css", "opencv", "computer vision",
    "data science", "data analysis", "feature engineering",
    "model evaluation", "random forest", "logistic regression",
    "naive bayes", "svm", "neural network", "regression",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text: str) -> list:
    found = set()

    for skill in TECH_SKILLS:
        if skill in text:
            found.add(skill)

    doc = nlp(text[:5000])
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG"):
            val = ent.text.lower().strip()
            if val in TECH_SKILLS:
                found.add(val)

    return sorted(list(found))


def extract_education(text: str) -> list:
    edu_keywords = [
        "b.tech", "btech", "b.e", "be", "m.tech", "mtech",
        "bachelor", "master", "mba", "bsc", "msc", "phd",
        "computer science", "information technology",
        "data science", "artificial intelligence", "electronics",
    ]
    return [kw for kw in edu_keywords if kw in text]


# ✅ FINAL UPDATED FUNCTION (IMPORTANT)
def process_file(filepath: str = None, text: str = None) -> dict:
    """
    Works for BOTH:
    - File input → process_file("data/resume.txt")
    - Text input → process_file(text="your resume text")
    """

    if text is not None:
        raw = text
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

    clean = clean_text(raw)

    return {
        "raw": raw,
        "clean": clean,
        "skills": extract_skills(clean),
        "education": extract_education(clean),
        "word_count": len(clean.split()),
    }


# Optional (kept for backward compatibility)
def process_text(text: str) -> dict:
    return process_file(text=text)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  STEP 1 TEST — preprocess.py")
    print("="*55)

    for label, path in [("RESUME", "data/resume.txt"), ("JOB DESC", "data/job.txt")]:
        print(f"\n--- {label} ---")
        r = process_file(path)
        print(f"  Word count : {r['word_count']}")
        print(f"  Skills     : {r['skills']}")
        print(f"  Education  : {r['education']}")

    print("\npreprocess.py — OK")