import fitz 
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

# Loading GloVe model
glove_vectors = KeyedVectors.load_word2vec_format("gloveEmbedding\glove.6B.100d.txt.w2v", binary=False)

# Text Extraction
def extract_text_from_pdf(pdf_file):
    # pages = fitz.open(pdf_file)
    pages = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""   
    for page_num in range(len(pages)):
        page = pages.load_page(page_num)
        text += page.get_text()
    print(text[:100])
    return text

# Preprocessing
def get_preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word not in stop_words]
    return tokens

# sentence embedding
def get_sentence_embedding(tokens, glove_vectors):
    embeddings = [glove_vectors[word] for word in tokens if word in glove_vectors]
    sentence_embedding = np.mean(embeddings, axis=0)
    return sentence_embedding
    
# closest sentences
def find_closest_sentences(question, text, glove_vectors, n=2):
    question_tokens = get_preprocess_text(question)
    question_embedding = get_sentence_embedding(question_tokens, glove_vectors)
    
    sentences = sent_tokenize(text)
    sentence_embeddings = []
    for sentence in sentences:
        sentence_tokens = get_preprocess_text(sentence)
        sentence_embedding = get_sentence_embedding(sentence_tokens, glove_vectors)
        sentence_embeddings.append((sentence, sentence_embedding))
    # print(len(sentence_embeddings))
    # print(question_embedding)

    similarities = [(sentence, cosine_similarity([question_embedding], [s_embedding])[0][0]) for sentence, s_embedding in sentence_embeddings]
    
    sorted_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)
    closest_sentences = [sentence for sentence, _ in sorted_sentences[:n]]
    
    return " ".join(closest_sentences)
