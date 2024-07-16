import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from gensim.models import KeyedVectors
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import pickle



#----------------------------------word2vec_model multiprocesamiento--------------------------------
# Función para inicializar el pool con el modelo
def iniciar_pool(model_path):
    print("Cargando Word2Vec model...")
    global word2vec_model
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word2Vec model cargado exitosamente")

def generate_word2vec_embeddings(text):
    tokens = text.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def parallel_generate_embeddings(texts,model_path):
    print("inicia funcion")
    num_cores = cpu_count()
    num_cores = max(1, num_cores - 2)  # Usa 2 núcleos menos que el total disponible    pool = Pool(processes=num_cores, initializer=iniciar_pool, initargs=(model_path,))

    resultados = []
    with tqdm(total=len(texts)) as pbar:    
        for resultado in pool.imap(generate_word2vec_embeddings, texts):
            resultados.append(resultado)
            pbar.update()

    pool.close()
    pool.join()
    print("termina funcion")
    return np.array(resultados)

#--------------------------fin word2vec_model multiprocesamiento----------------------------------


#-------------------------------------BERT multiprocesamiento----------------------------------

# Función para inicializar el pool con el modelo y el tokenizer
def init_pool(model_name):
    global tokenizer, model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertModel.from_pretrained(model_name)
    
def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)   
    return outputs.last_hidden_state[:, 0, :]  # Usar la representación del token [CLS]

def paralelo_bert(texts,model_name):
    print("inicia funcion")
    num_cores = cpu_count()
    #num_cores = max(1, num_cores - 2)  # Usa 2 núcleos menos que el total disponible
    print(f"Usando {num_cores} cores")
    pool = Pool(processes=num_cores, initializer=init_pool, initargs=(model_name,))

    resultados = []
    with tqdm(total=len(texts)) as pbar:    
        for resultado in pool.imap(generate_bert_embeddings, texts):
            resultados.append(resultado)
            pbar.update()

    pool.close()
    pool.join()
    print("termina funcion")
    return np.array(resultados).transpose(0,2,1)

if __name__ == "__main__":
    # Load data
    wine_df = pd.read_csv('./data/winemag-data_first150k.csv')
    corpus = wine_df['description']
    print("Datos cargados exitosamente")
    #word2vec_model = KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)
    #word2vec_embeddings = parallel_generate_embeddings(corpus,'word2vec_model.bin')  
    # Output results
    #print("Word2Vec Embeddings:", word2vec_embeddings)
    #print("Word2Vec Shape:", word2vec_embeddings.shape)

    # Load pre-trained BERT model and tokenizer
    bert_embeddings = paralelo_bert(corpus,'bert-base-uncased')

    with open('bert_embeddings.pkl', 'wb') as f:
        pickle.dump(bert_embeddings, f)

    # Guardar la variable en un archivo npy
    np.save('bert_embeddings.npy', bert_embeddings)

    print("Bert Shape:", bert_embeddings.shape)

 



