#  AI-Powered Resume Screening & Job Matching System

An AI/ML-based project that automatically screens resumes and matches them with job descriptions using Natural Language Processing (NLP) techniques like TF-IDF and Cosine Similarity.

---

##  Features

*  Upload and extract text from PDF resumes
*  Clean and preprocess text data
*  Match resumes with job descriptions using similarity scoring
*  Rank candidates based on relevance score
*  Interactive web interface using Streamlit
*  Store and manage data using SQLite database

---

##  Tech Stack

* Python 
* Streamlit (Frontend UI)
* Scikit-learn (ML algorithms)
* TF-IDF Vectorization
* Cosine Similarity
* pypdf (PDF text extraction)
* SQLite (Database)
* NLP (Natural Language Processing)

---

##  Project Structure

```
project-folder/
│
├── app.py                # Streamlit application
├── model.py             # ML logic (TF-IDF, similarity)
├── prepare_data.py      # Data preprocessing
├── main.py              # Entry point
├── database.db          # SQLite database
└── README.md            # Project documentation
```

---

##  How It Works

1. User uploads resume (PDF format)
2. Text is extracted using `pypdf`
3. Text is cleaned and preprocessed
4. TF-IDF converts text into numerical vectors
5. Cosine similarity compares resume with job description
6. System ranks resumes based on relevance score

---

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resume-screening-ai.git
cd resume-screening-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

##  Example Use Case

* HR uploads multiple resumes
* System automatically ranks candidates
* Saves time in manual screening
* Improves hiring efficiency

---

## Future Improvements

* Integration with LLMs (GPT-based evaluation)
* API-based deployment (FastAPI)
* Advanced semantic search using embeddings
* Multi-job matching system

---

##  Author

**Sanjeev Kumar Yadav**
MCA Student | AI/ML Enthusiast

---

##  Note

This project is built for learning and demonstration purposes of NLP-based document similarity systems.
