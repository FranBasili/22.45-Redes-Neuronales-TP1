import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import tensorflow as tf

def plot_scatter(dataframe): 
  plt.figure(figsize=(10,25))
  for i in range(10):
    plt.subplot(5,1,1)
    plt.scatter(x=dataframe["length"][train_labels==i], 
                y=dataframe["slant"][train_labels==i],  
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("largo")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,2)
    plt.scatter(x=dataframe["thickness"][train_labels==i], 
                y=dataframe["slant"][train_labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("grosor")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,3)
    plt.scatter(x=dataframe["width"][train_labels==i], 
                y=dataframe["slant"][train_labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("Ancho")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,4)
    plt.scatter(x=dataframe["height"][train_labels==i], 
                y=dataframe["slant"][train_labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("Alto")
    plt.ylabel("inclinacion")

  for i in range(10):
    plt.subplot(5,1,5)
    plt.scatter(x=dataframe["area"][train_labels==i],
                y=dataframe["slant"][train_labels==i], 
                color="C"+str(i), label="Clase "+str(i),
                alpha=0.5)
    plt.legend()
    plt.xlabel("Area")
    plt.ylabel("inclinacion")
    
  plt.show()