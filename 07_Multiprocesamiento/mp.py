from multiprocessing import Pool, cpu_count
import sys
from tqdm import tqdm
import pandas as pd
# Cálculo secuencial del factorial
def calcular_factorial(numero):
    factorial = 1
    for i in range(1, numero + 1):
        factorial *= i
    return factorial



if __name__ == "__main__":
    # Aumentar el tamaño máximo de los enteros
    sys.setrecursionlimit(100000000)
    
    # Definir el rango de números para calcular sus factoriales
    numeros = range(1, 5000)
    
    #obetener cores de mi pc actual
    cores = cpu_count()
    print(f"Usando {cores} cores")
    # Crear un Pool de procesos
    pool = Pool(processes=cores)
    resultados = []  # Lista para almacenar los resultados

    #pool.imap: Ideal para actualizar la barra de progreso en tiempo real, 
    #           ya que produce resultados parciales a medida que las tareas se completan.
    
    # pool.map: Devuelve todos los resultados a la vez, lo que significa que la 
    #           barra de progreso se actualiza al final.



    # Usar tqdm para mostrar la barra de progreso
    with tqdm(total=len(numeros)) as pbar:
        # Usar imap para obtener resultados de forma asíncrona
        for resultado in pool.imap(calcular_factorial, numeros):
            resultados.append(resultado)  # Guardar cada resultado en la lista
            pbar.update()

#no se actualiza la barra de progreso en tiempo real con map

    # with tqdm(total=len(numeros)) as pbar:
    #     # Usar pool.map y actualizar la barra de progreso
    #     resultados = pool.map(calcular_factorial, numeros)
    #     pbar.update(len(numeros))  # Actualizar la barra de progreso al final

    # Cerrar el Pool de procesos
    pool.close()
    pool.join()
    df = pd.DataFrame({
        'Numero': numeros,
        'Factorial': resultados
    })
    df.to_csv('resultados_factorial.csv', index=False)