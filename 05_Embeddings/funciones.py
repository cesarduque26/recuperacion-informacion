import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from gensim.models import KeyedVectors
# Load pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)

print("Word2Vec model cargado exitosamente")

def generate_word2vec_embeddings(texts):
    embeddings = []
    for text in tqdm(texts, desc="Generating Word2Vec Embeddings"):
        tokens = text.lower().split()
        word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))
    return np.array(embeddings)

def parallel_generate_embeddings(texts):
    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        embeddings = pool.map(generate_word2vec_embeddings, np.array_split(texts, num_cores))
    return np.concatenate(embeddings)