import streamlit as st
from preprocess import process_file
from tfidf_matcher import tfidf_match_score
from embedding_matcher import embedding_match_score

st.set_page_config(page_title="AI Resume Screener", layout="centered")

st.title("📄 AI Resume Screener")
st.write("Compare your resume with a job description using AI")

# Input fields
resume_text = st.text_area("Paste your Resume", height=200)
job_text = st.text_area("Paste Job Description", height=200)

if st.button("Analyze"):
    if not resume_text or not job_text:
        st.warning("Please enter both resume and job description")
    else:
        # Process text
        resume_data = process_file(None, text=resume_text)
        job_data = process_file(None, text=job_text)

        # Scores
        tfidf = tfidf_match_score(resume_text, job_text)
        emb = embedding_match_score(resume_text, job_text)

        combined = round((tfidf["score_percent"] * 0.4 + emb["score_percent"] * 0.6), 2)

        # Display scores
        st.subheader("📊 Scores")
        st.write(f"**TF-IDF Score:** {tfidf['score_percent']}%")
        st.write(f"**Embedding Score:** {emb['score_percent']}%")
        st.write(f"**Combined Score:** {combined}%")

        # Skill analysis
        st.subheader("🧠 Skill Analysis")

        st.write("✅ Matched Skills:")
        st.write(", ".join(tfidf["matched_skills"]))

        st.write("❌ Missing Skills:")
        st.write(", ".join(tfidf["missing_skills"]))

        # Recommendation
        st.subheader("💡 Recommendation")

        if combined >= 75:
            st.success("Strong profile! Apply now.")
        elif combined >= 50:
            st.warning("Moderate match. Improve a few skills.")
        else:
            st.error("Low match. Work on missing skills.")