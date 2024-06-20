from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'gloveEmbedding\glove.6B.100d.txt'  # Adjust path as needed
word2vec_output_file = 'gloveEmbedding\glove.6B.100d.txt.w2v'
glove2word2vec(glove_input_file, word2vec_output_file)
