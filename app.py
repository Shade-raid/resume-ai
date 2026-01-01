import streamlit as st
import PyPDF2
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# ------------------ NLTK Setup ------------------
nltk.download("stopwords", quiet=True)

# ------------------ FUNCTIONS ------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors)[0][1]
    return round(score * 100, 2)

def extract_keywords(text, n=20):
    vectorizer = TfidfVectorizer(max_features=n, stop_words="english")
    vectorizer.fit([text])
    return vectorizer.get_feature_names_out()

# ------------------ STREAMLIT UI ------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide",
    page_icon="📄"
)

# ------------------ HEADER ------------------
st.markdown(
    """
    <div style="background-color:#4B8BBE;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">📄 AI Resume Analyzer</h1>
        <p style="color:white;text-align:center;font-size:16px;">
        Compare your resume against a job description and see skill matches.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ------------------ INPUT SECTION ------------------
with st.container():
    st.subheader("Upload & Job Description")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        resume_file = st.file_uploader("📁 Upload Resume (PDF)", type=["pdf"])
        
    with col2:
        job_description = st.text_area(
            "📝 Paste Job Description",
            placeholder="Paste the job description here..."
        )

# ------------------ ANALYSIS BUTTON ------------------
analyze_btn = st.button("Analyze Resume", type="primary")

# ------------------ RESULTS SECTION ------------------
if analyze_btn:
    if resume_file and job_description:
        with st.spinner("🔍 Analyzing resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description)

            score = calculate_similarity(resume_clean, jd_clean)

            resume_keywords = set(extract_keywords(resume_clean))
            jd_keywords = set(extract_keywords(jd_clean))

            matched = resume_keywords & jd_keywords
            missing = jd_keywords - resume_keywords

        st.success("✅ Analysis Complete!")

        # ---------- MATCH SCORE ----------
        st.metric(label="Match Score (%)", value=f"{score}%")

        # ---------- PROGRESS BAR ----------
        st.progress(score / 100)

        st.write("---")

        # ---------- KEYWORDS SECTION ----------
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✔ Matching Keywords")
            if matched:
                st.markdown(", ".join(sorted(matched)))
            else:
                st.info("No strong keyword matches found.")

        with col2:
            st.subheader("❌ Missing Keywords")
            if missing:
                st.markdown(", ".join(sorted(missing)))
            else:
                st.info("No missing skills detected.")

        st.write("---")

        # ---------- OPTIONAL: TOP RESUME KEYWORDS ----------
        st.subheader("📌 Top Keywords in Resume")
        st.write(", ".join(sorted(resume_keywords)))
    else:
        st.warning("Please upload a resume and paste a job description.")
