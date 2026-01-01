import streamlit as st
import PyPDF2
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Download once (Streamlit Cloud compatible)
nltk.download("stopwords", quiet=True)

# ================= FUNCTIONS =================
def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def similarity_score(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(vectors)[0][1]
    return round(score * 100, 2)

def extract_keywords(text, n=20):
    tfidf = TfidfVectorizer(max_features=n)
    tfidf.fit([text])
    return tfidf.get_feature_names_out()

# ================= UI =================
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer (Offline)")
st.caption("No APIs • No Internet • Fully Local NLP")

resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume_file and job_desc:
        resume_text = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_desc)

        score = similarity_score(resume_clean, jd_clean)
        resume_keywords = set(extract_keywords(resume_clean))
        jd_keywords = set(extract_keywords(jd_clean))

        matched = resume_keywords & jd_keywords
        missing = jd_keywords - resume_keywords

        st.success(f"✅ Match Score: {score}%")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✔ Matching Keywords")
            st.write(", ".join(sorted(matched)) if matched else "None")

        with col2:
            st.subheader("❌ Missing Keywords")
            st.write(", ".join(sorted(missing)) if missing else "None")
    else:
        st.warning("Upload resume and paste job description.")
