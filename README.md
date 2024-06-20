# Overview
This project is a PDF question answering (QA) system that leverages natural language processing (NLP) techniques to extract relevant sections from PDF documents in response to user queries. The system uses GloVe embeddings converted to a Word2Vec-compatible format for sentence similarity calculations and provides an interactive web interface built with Streamlit.
# Installations
- download the gloVe Embedding from the [official website](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip) , and save it in the gloveEmbedding folder
- run the g_to_w2v.py file
To Run the Pogramme in your system you need to: 
- clone the repo 
```bash
git clone github link
cd repo_name
```
- install the required packages
```bash
pip intall -r requirments.txt
```
# Usage
- Run the application
```bash
streamlit run app.py
```
- upload the pdf file 
- ask the question
- enter to get the answer

# Project Structure
- logic.py: Contains the core logic for processing PDFs, extracting text, converting embeddings, and calculating sentence similarity.
- app.py: Contains the Streamlit code to create the web interface for the application.
- gloveEmbedding:
    - g_to_w2v.py: file converting glove embedding to compatible form
    - glove.6B.100d.txt:glove embedding
    - glove.6B.100d.txt.w2v: The Word2Vec compatible form of glove embedding 
# Dependencies
* Python 3.7+
* Gensim
* PyMuPDF
* NLTK
* Scikit-learn
* Streamlit
