import streamlit as st
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Text Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Keyword Extraction
def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Match keywords
def match_keywords(jd_keywords, resume_text):
    matched = [word for word in jd_keywords if word in resume_text]
    missing = [word for word in jd_keywords if word not in resume_text]
    match_score = (len(matched) / len(jd_keywords)) * 100
    return matched, missing, match_score

# Streamlit App Layout
st.title("📊 AI-Powered Resume Keyword Matcher")

st.write("Upload your Job Description and Resume as text to see how well they match.")

# Text Input Areas
jd_text = st.text_area("📄 Enter Job Description Text")
resume_text = st.text_area("📝 Enter Resume Text")

if st.button("🔍 Match Now"):
    if jd_text and resume_text:
        jd_clean = preprocess_text(jd_text)
        resume_clean = preprocess_text(resume_text)
        
        jd_keywords = extract_keywords(jd_clean)
        matched, missing, score = match_keywords(jd_keywords, resume_clean)
        
        st.success(f"✅ Match Score: {score:.2f}%")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([len(matched), len(missing)], labels=['Matched', 'Missing'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(jd_keywords))
        st.subheader("📌 Job Description Keywords WordCloud")
        st.image(wordcloud.to_array())
        
        # Show matched and missing keywords
        st.subheader("✅ Matched Keywords")
        st.write(", ".join(matched) if matched else "No Matches Found.")
        
        st.subheader("❌ Missing Keywords")
        st.write(", ".join(missing) if missing else "None. Perfect Match!")
        
    else:
        st.warning("Please enter both Job Description and Resume Text.")


