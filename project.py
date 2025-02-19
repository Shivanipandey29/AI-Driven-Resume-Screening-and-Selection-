import streamlit as st
from PyPDF2 import PdfReader
import re
import spacy
import subprocess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Ensure spaCy model is downloaded dynamically
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Ensure NLTK stopwords are available
nltk.download('stopwords')
try:
    stopwords_list = set(stopwords.words('english'))
except:
    stopwords_list = set()  # Fallback in case of error

def extract_skills_from_resume(text):
    return extract_skills(text)

def extract_skills(text):
    skills_start = text.find('Skills')
    skills_end = text.find('SKILLS') if text.find('SKILLS') != -1 else len(text)
    skills_text = text[skills_start:skills_end].strip()

    doc = nlp(skills_text)
    skill_sentences = [sent.text.strip() for sent in doc.sents if any(token.pos_ in ['NOUN', 'VERB', 'ADJ'] for token in sent)]
    
    return ' '.join(skill_sentences) if skill_sentences else 'N/A'

def clean_and_tokenize_skills(skill_string):
    if isinstance(skill_string, str):
        cleaned_string = re.sub(r'[/\\]+|\n|\s+', ' ', skill_string)
        return word_tokenize(cleaned_string)
    return []

def fetch_my_score(resume_text):
    all_skills = extract_skills_from_resume(resume_text)
    cleaned_skills = clean_and_tokenize_skills("".join(all_skills).lower())

    new_top_skills = [
        'algorithms', 'analytical', 'analytics', 'artificial intelligence', 'aws',
        'azure', 'big data', 'business intelligence', 'c++', 'cloud', 'communication',
        'data analysis', 'data science', 'deep learning', 'docker', 'excel',
        'machine learning', 'matplotlib', 'natural language processing', 'neural networks',
        'numpy', 'pandas', 'power bi', 'predictive modeling', 'python', 'scikit-learn',
        'sql', 'statistics', 'tableau', 'tensorflow'
    ]

    matched_skills = [skill for skill in cleaned_skills if skill in new_top_skills]
    count = len(matched_skills)

    score = (
        9 if count > 20 else
        8 if count >= 15 else
        7 if count >= 10 else
        6 if count >= 6 else
        1 if count < 2 else
        4
    )

    remaining_skills = [skill for skill in new_top_skills if skill not in matched_skills]

    return score, matched_skills, remaining_skills

def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

def text_to_vector(text):
    words = preprocess_text(text)
    word_vectors = [token.vector for token in nlp(" ".join(words)) if token.has_vector]
    return sum(word_vectors) / len(word_vectors) if word_vectors else None

def remove_un(text):
    if isinstance(text, str):
        return " ".join(["".join(e for e in word if e.isalnum()).lower() for word in text.split() if word.lower() not in stopwords_list])
    elif isinstance(text, list):
        return [" ".join(["".join(e for e in word if e.isalnum()).lower() for word in t.split() if word.lower() not in stopwords_list]) for t in text if isinstance(t, str)]
    return None

def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0] - 0.10 if vector1 is not None and vector2 is not None else None

def main():
    st.title("AI Driven Resume Screening and Scanning")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    job_description = st.text_area("Enter Job Description")

    if uploaded_file is not None:
        with uploaded_file:
            resume_text = "".join([page.extract_text() for page in PdfReader(uploaded_file).pages])

        score, mentioned_skills, not_mentioned_skills = fetch_my_score(resume_text)

        st.header("Resume Score and Skills")
        st.write(f"Score: {score}")
        st.write("Skills Mentioned in Resume:", mentioned_skills)
        st.write("Skills Not Mentioned in Resume (Recommendations):", not_mentioned_skills)

        resume_vector = text_to_vector(remove_un(resume_text))
        job_vector = text_to_vector(remove_un(job_description))
        similarity = calculate_cosine_similarity(resume_vector, job_vector)

        st.header("Similarity with Job Description")
        st.write(f"Similarity: {similarity}" if similarity is not None else "Unable to calculate similarity.")

if __name__ == "__main__":
    main()
