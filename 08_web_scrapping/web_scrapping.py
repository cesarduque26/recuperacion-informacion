import pickle
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np

headers = {
    "User-Agent":"My Phyton App"
}
def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').replace('\xa0', ' ').replace(';', ' ').strip()


def process_link(link):
    try:
        response = requests.get(link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        try:
            titulo = soup.find('title').text
        except Exception as e:
            titulo = ''

        try:
            descripcion = clean_text(soup.find('p', class_='article-subheading type--dog').text) 

        except Exception as e:
            descripcion = ''

        try:
            listas = soup.find_all('li', class_='mm-recipes-structured-ingredients__list-item')
            ingredientes = ''
            for lista in listas:
                spans = lista.find_all('span')
                for span in spans:
                    ingredientes += span.text + ' '
                ingredientes += ' | '
        except Exception as e:
            ingredientes = ''

        try:
            parrafos = soup.find_all('p', class_='comp mntl-sc-block mntl-sc-block-html')
            receta = ''
            for parrafo in parrafos:
                parrafo = clean_text(parrafo.text)
                receta += parrafo+ ' '
                receta += ' | '
        except Exception as e:
            receta = ''

        return {
            'Titulo': titulo,
            'Descripcion': descripcion,
            'Ingredientes': ingredientes.strip(),
            'Receta': receta.strip(),
            'Link': link
        }

    except Exception as e:
        print(f"Error procesando el enlace {link}: {e}")
        return {
            'Titulo': '',
            'Descripcion': '',
            'Ingredientes': '',
            'Receta': '',
            'Link': link
        }
    
if __name__ == '__main__':
        print('cargando links...')
        link_total = pickle.load(open('/data/link_total.pkl', 'rb'))
        print(f'links cargados: {len(link_total)}')
        # Número de procesos a utilizar (usaremos el número de núcleos lógicos de CPU)
        num_processes = cpu_count()

        # Usamos Pool para crear un grupo de procesos y aplicar multiprocessing
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_link, link_total), total=len(link_total), desc='Procesando'))

        # Convertir los resultados en un DataFrame
        corpus = pd.DataFrame(results)
        corpus.replace('', np.nan, inplace=True)
        corpus = corpus.dropna()
        # Guardar el DataFrame en un archivo CSV
        corpus.to_csv('corpus_multiprocessing.csv', index=False, encoding='utf-8')

        # Mostrar el DataFrame
        print(corpus)    