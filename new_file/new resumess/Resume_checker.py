import streamlit as st
from PyPDF2 import PdfReader
import re
import spacy
from nltk.tokenize import word_tokenize

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import spacy
from PyPDF2 import PdfReader
import pandas as pd
import re


nlp = spacy.load('en_core_web_sm')

# Load spaCy model
nlp = spacy.load("en_core_web_md")

stopwords = stopwords.words('english')

def extract_skills_from_resume(text):
    # Extract Skills
    skills = extract_skills(text)
    return skills

def extract_skills(text):
    skills_start = text.find('Skills')
    skills_end = text.find('SKILLS') if text.find('SKILLS') != -1 else len(text)
    skills_text = text[skills_start:skills_end].strip()

    # Process the skills text with spaCy
    doc = nlp(skills_text)

    # Extract lines or sentences containing skill-related information
    skill_sentences = [sent.text.strip() for sent in doc.sents if any(token.pos_ in ['NOUN', 'VERB', 'ADJ'] for token in sent)]

    # Combine the extracted lines into a single string
    skills = ' '.join(skill_sentences)

    return skills if skills else 'N/A'

def clean_and_tokenize_skills(skill_string):
    if isinstance(skill_string, str):
        # Remove '/', '\n', extra spaces
        cleaned_string = re.sub(r'[/\\]+|\n|\s+', ' ', skill_string)

        # Tokenize the skills using nltk's word_tokenize
        tokenized_skills = word_tokenize(cleaned_string)

        return tokenized_skills
    else:
        # Return an empty list if the input is not a string
        return []

def fetch_my_score(resume_text):
    all_skills = extract_skills_from_resume(resume_text)
    all_skills = "".join(all_skills)
    cleaned_skills = clean_and_tokenize_skills(all_skills)

    new_top_skills = ['algorithms', 'analytical', 'analytical skills', 'analytics', 'artificial intelligence', 'aws',
                      'azure', 'beautiful soup', 'big data', 'business intelligence', 'c++', 'cloud', 'coding',
                      'communication', 'computer science', 'computer vision', 'css', 'data analysis', 'data analyst',
                      'data analytics', 'data collection', 'data management', 'data mining', 'data modeling',
                      'data quality', 'data science', 'data scientist', 'data structures', 'data visualization',
                      'deep learning', 'docker', 'excel', 'financial services', 'flask', 'forecasting', 'git', 'hadoop',
                      'html', 'java', 'javascript', 'keras', 'logistic regression', 'machine learning', 'management',
                      'matplotlib', 'natural language processing', 'neural networks', 'nlp', 'numpy', 'pandas', 'power bi',
                      'predictive modeling', 'programming', 'project management', 'python', 'pytorch', 'r', 'react', 'sas',
                      'scikit', 'scipy', 'seaborn', 'selenium', 'spark', 'sql', 'statistical modeling', 'statistics',
                      'tableau', 'tensorflow', 'testing', 'web scraping']

    count = 0
    skills = []

    for skill in cleaned_skills:
        skill = skill.lower()
        if skill in new_top_skills:
            skills.append(skill)
            count += 1

    if count > 20:
        final_score = 9
    elif 15 <= count < 20:
        final_score = 8
    elif 10 <= count <= 14:
        final_score = 7
    elif 6 <= count <= 9:
        final_score = 6
    elif count < 2:
        final_score = 1
    else:
        final_score = 4

    remaining_skills = [i for i in new_top_skills if i not in skills]

    return final_score, skills, remaining_skills

def main():
    st.title("Resume Score and Skills Checker")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    job_description = st.text_area("Enter Job Description")

    if uploaded_file is not None:
        # Read the uploaded PDF file
        with uploaded_file:
            resume_text = ""
            pdf_reader = PdfReader(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                resume_text += pdf_reader.pages[page_num].extract_text()

        # Fetch the score and skills
        score, mentioned_skills, not_mentioned_skills = fetch_my_score(resume_text)

        # Display the results
        st.header("Resume Score and Skills")
        st.write(f"*Score:* {score}")
        st.write("*Skills Mentioned in Resume:*")
        st.write(mentioned_skills)
        st.write("*Skills Not Mentioned in Resume (Recommendations):*")
        st.write(not_mentioned_skills)

        # Calculate Similarity
        resume_vector = text_to_vector(remove_un(resume_text))
        job_vector = text_to_vector(remove_un(job_description))
        similarity = calculate_cosine_similarity(resume_vector, job_vector)

        st.header("Similarity with Job Description")
        st.write(f"Similarity: {similarity}")

if __name__ == "__main__":
    main()
