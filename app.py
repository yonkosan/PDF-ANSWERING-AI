import streamlit as st
from logic import extract_text_from_pdf, find_closest_sentences, glove_vectors

st.title("PDF Question Answering AI")
st.write("Upload a PDF file and ask a question to get the most relevant answer.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("PDF text extracted successfully!")

    question = st.text_input("Enter your question:")
    
    if question:

        answer = find_closest_sentences(question, pdf_text, glove_vectors)
        st.write("Answer:", answer)
