import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import tensorflow as tf

def plot_scatter(dataframe, labels): 
  plt.figure(figsize=(10,25))
  for i in range(10):
    plt.subplot(5,1,1)
    plt.scatter(x=dataframe["length"][labels==i], 
                y=dataframe["slant"][labels==i],  
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("largo")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,2)
    plt.scatter(x=dataframe["thickness"][labels==i], 
                y=dataframe["slant"][labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("grosor")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,3)
    plt.scatter(x=dataframe["width"][labels==i], 
                y=dataframe["slant"][labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("Ancho")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,4)
    plt.scatter(x=dataframe["height"][labels==i], 
                y=dataframe["slant"][labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("Alto")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,5)
    plt.scatter(x=dataframe["area"][labels==i],
                y=dataframe["slant"][labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("Area")
    plt.ylabel("inclinacion")
    
  plt.show()

def plot_embeddings(vectors):
  for i in range(len(vectors)):
    plt.scatter(x=vectors[i][0], y=vectors[i][1], label=str(i))
  
  plt.title("Embeddings")
  plt.legend()
  plt.grid()
  plt.show()