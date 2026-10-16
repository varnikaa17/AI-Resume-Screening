
import streamlit as st
import pandas as pd
import joblib
import re
import fitz
from docx import Document
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "models" / "similarity_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "models" / "similarity_vectorizer.pkl")
    return model, vectorizer

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text_parts = []
        with fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        ) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    if name.endswith(".docx"):
        doc = Document(uploaded_file)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    return ""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_skills(text):
    skills = [
        "python", "java", "javascript", "c++", "sql",
        "html", "css", "react", "node.js", "rest api",
        "machine learning", "deep learning", "pandas",
        "numpy", "scikit-learn", "tensorflow", "pytorch",
        "nlp", "git", "docker", "aws", "linux",
        "networking", "wireshark", "spring boot",
        "mongodb", "data analysis", "statistics"
    ]

    text = clean_text(text)
    return sorted({
        skill for skill in skills
        if re.search(
            r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])",
            text
        )
    })

st.title("📄 AI Resume Screening System")
st.write(
    "Compare a resume with a job description and inspect "
    "job-related skill matches."
)

st.info(
    "This is a prototype. Its model was trained on synthetic data, "
    "so the score is experimental and must not be used as a hiring decision."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload resume (PDF or DOCX)",
        type=["pdf", "docx"]
    )

with col2:
    job_description = st.text_area(
        "Paste job description",
        height=220,
        placeholder="Enter the responsibilities and required skills..."
    )

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a PDF or DOCX resume.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        try:
            resume_text = extract_text(uploaded_file)

            if len(resume_text.strip()) < 30:
                st.error(
                    "Could not extract enough text. "
                    "If this is a scanned PDF, OCR is needed."
                )
            else:
                model, vectorizer = load_model()

                resume_clean = clean_text(resume_text)
                job_clean = clean_text(job_description)

                resume_vector = vectorizer.transform([resume_clean])
                job_vector = vectorizer.transform([job_clean])

                similarity = float(
                    cosine_similarity(resume_vector, job_vector)[0][0]
                )

                combined_text = [resume_clean + " " + job_clean]
                combined_vector = vectorizer.transform(combined_text)

                features = __import__("scipy").sparse.hstack(
                    [combined_vector, [[similarity]]]
                ).tocsr()

                score = float(model.predict_proba(features)[0][1])

                resume_skills = set(extract_skills(resume_text))
                job_skills = set(extract_skills(job_description))

                matched = sorted(resume_skills & job_skills)
                missing = sorted(job_skills - resume_skills)

                st.divider()
                st.subheader("Analysis Results")

                metric1, metric2 = st.columns(2)
                metric1.metric("Model match score", f"{score * 100:.1f}%")
                metric2.metric("Text similarity", f"{similarity * 100:.1f}%")

                st.caption(
                    "These scores are experimental outputs, not calibrated "
                    "probabilities of job success."
                )

                left, right = st.columns(2)

                with left:
                    st.subheader("✅ Skills found in both")
                    if matched:
                        st.write(", ".join(matched))
                    else:
                        st.write("No listed skills matched.")

                with right:
                    st.subheader("📝 Job skills not found")
                    if missing:
                        st.write(", ".join(missing))
                    else:
                        st.write("All detected job skills were found.")

                with st.expander("View extracted resume text"):
                    st.text(resume_text)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
