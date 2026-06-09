"""
predict.py — Improved Version
"""

import argparse
from colorama import Fore, Style, init
from tabulate import tabulate

from preprocess        import process_file
from tfidf_matcher     import tfidf_match_score, skill_match_percent
from embedding_matcher import embedding_match_score
from faiss_ranker      import rank_jobs

init(autoreset=True)


# ----------------------------
# Helpers
# ----------------------------
def color_score(score: float) -> str:
    if score >= 75:
        return Fore.GREEN + f"{score}%" + Style.RESET_ALL
    elif score >= 50:
        return Fore.YELLOW + f"{score}%" + Style.RESET_ALL
    else:
        return Fore.RED + f"{score}%" + Style.RESET_ALL


def print_header(title: str):
    print("\n" + Fore.CYAN + "="*60)
    print(f"  {title}")
    print("="*60 + Style.RESET_ALL)


def final_interpretation(score: float) -> str:
    if score >= 75:
        return "Excellent match — apply confidently ✅"
    elif score >= 50:
        return "Good match — minor improvements needed 👍"
    elif score >= 30:
        return "Moderate match — improve key skills ⚠️"
    else:
        return "Low match — significant upskilling required ❌"


# ----------------------------
# MODE: SCORE
# ----------------------------
def mode_score(resume_path: str, job_path: str):
    print_header("RESUME SCREENER — MATCH SCORE REPORT")

    resume_data = process_file(resume_path)
    job_data    = process_file(job_path)

    print(f"\n  Resume words    : {resume_data['word_count']}")
    print(f"  Job desc words  : {job_data['word_count']}")

    print(f"\n  Running TF-IDF analysis...")
    tfidf = tfidf_match_score(resume_data["raw"], job_data["raw"])

    print(f"  Running embedding analysis...")
    emb = embedding_match_score(resume_data["raw"], job_data["raw"])

    combined = round((tfidf["score_percent"] * 0.4 + emb["score_percent"] * 0.6), 2)
    interpretation = final_interpretation(combined)

    print_header("SCORES")
    print(tabulate([
        ["TF-IDF Score",        color_score(tfidf["score_percent"])],
        ["Embedding Score",     color_score(emb["score_percent"])],
        ["Combined Score",      color_score(combined)],
        ["Final Verdict",       interpretation],
    ], tablefmt="simple"))

    coverage = skill_match_percent(tfidf["matched_skills"], tfidf["job_skills"])

    print_header("SKILL ANALYSIS")
    print(tabulate([
        ["Skills required", len(tfidf["job_skills"])],
        ["Skills matched",  len(tfidf["matched_skills"])],
        ["Skills missing",  len(tfidf["missing_skills"])],
        ["Coverage",        color_score(coverage)],
    ], tablefmt="simple"))

    # Matched
    if tfidf["matched_skills"]:
        print(f"\n{Fore.GREEN}Matched Skills:{Style.RESET_ALL}")
        print(", ".join(tfidf["matched_skills"]))

    # Missing
    if tfidf["missing_skills"]:
        print(f"\n{Fore.RED}Missing Skills:{Style.RESET_ALL}")
        print(", ".join(tfidf["missing_skills"]))

    # Suggestions
    print_header("RECOMMENDATION")

    if combined >= 75:
        print(Fore.GREEN + "Strong profile! Apply immediately.")
        print("Tip: Tailor your resume keywords to the job.")
    elif combined >= 50:
        print(Fore.YELLOW + "Good profile. Improve a few skills.")
        print("Tip: Add missing skills and projects.")
    elif combined >= 30:
        print(Fore.YELLOW + "Moderate match. Work on core skills.")
        print("Tip: Build 1–2 strong ML/NLP projects.")
    else:
        print(Fore.RED + "Low match. Significant improvement needed.")
        print("Tip: Learn fundamentals before applying.")

    print("\nDone.\n")


# ----------------------------
# MODE: RANK
# ----------------------------
def mode_rank(resume_path: str):
    print_header("RESUME SCREENER — JOB RANKING")

    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    sample_jobs = [
        ("Junior Data Scientist", "Python ML pandas numpy statistics"),
        ("NLP Engineer", "Python NLP transformers BERT deep learning"),
        ("ML Engineer", "Python TensorFlow PyTorch deployment FastAPI"),
        ("Frontend Developer", "HTML CSS JS React UI"),
        ("Database Admin", "SQL MySQL PostgreSQL indexing"),
    ]

    titles = [j[0] for j in sample_jobs]
    descs  = [j[1] for j in sample_jobs]

    ranked = rank_jobs(resume_text, descs, titles)

    table_data = [
        [f"#{r['rank']}", r["title"], color_score(r["score_percent"]), r["interpretation"]]
        for r in ranked
    ]

    print(tabulate(table_data,
                   headers=["Rank", "Role", "Score", "Verdict"],
                   tablefmt="simple"))

    top = ranked[0]
    print(f"\nBest role for you: {top['title']} ({top['score_percent']}%)\n")


# ----------------------------
# MODE: TIPS
# ----------------------------
def mode_tips(resume_path: str, job_path: str):
    print_header("RESUME IMPROVEMENT TIPS")

    resume_data = process_file(resume_path)
    job_data    = process_file(job_path)
    tfidf       = tfidf_match_score(resume_data["raw"], job_data["raw"])

    print(f"\nYour Skills  : {', '.join(tfidf['resume_skills'])}")
    print(f"Job Needs    : {', '.join(tfidf['job_skills'])}")

    print("\nAction Plan:")

    if tfidf["missing_skills"]:
        print(f"- Learn: {', '.join(tfidf['missing_skills'])}")

    if resume_data["word_count"] < 200:
        print("- Expand resume (add projects & experience)")

    if not resume_data["education"]:
        print("- Add education details")

    print("- Add measurable achievements (e.g., accuracy, results)")
    print("- Add GitHub links")

    print("\nDone.\n")


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["score","rank","tips"], default="score")
    parser.add_argument("--resume", default="data/resume.txt")
    parser.add_argument("--job", default="data/job.txt")
    args = parser.parse_args()

    if args.mode == "score":
        mode_score(args.resume, args.job)
    elif args.mode == "rank":
        mode_rank(args.resume)
    elif args.mode == "tips":
        mode_tips(args.resume, args.job)


if __name__ == "__main__":
    main()