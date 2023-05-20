from sklearn.metrics import confusion_matrix
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def buildModel(activation='relu', dropout=0.2, init= tf.keras.initializers.GlorotNormal(), batchNormalization=False):
  model=keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28,28))) 

  model.add(keras.layers.Dropout(dropout))

  model.add(keras.layers.Dense(512, activation=activation, kernel_initializer=init))
  model.add(keras.layers.Dropout(dropout))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add(keras.layers.Dense(128, activation=activation, kernel_initializer=init))
  model.add(keras.layers.Dropout(dropout))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add( keras.layers.Dense(128, activation=activation, kernel_initializer=init))
  model.add(keras.layers.Dropout(dropout))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add(keras.layers.Dense(128, activation=activation, kernel_initializer=init))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer=init))

  return model


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def showGraph(title="", epochs=20):
  plt.xticks(np.arange(1,CANT_EPOCHS+1), np.arange(1,CANT_EPOCHS+1), rotation=45)
  plt.ylabel("accuracy")
  plt.xlabel("epochs")
  plt.grid()
  plt.title(title)
  plt.legend()
  plt.show()

def DB_model():
  activation='relu'; dropout=0.2  
  init= tf.keras.initializers.GlorotNormal() 
  batchNormalization=True
  
  model=keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28,28))) 

  model.add(keras.layers.Dropout(dropout))

  model.add(keras.layers.Dense(512, activation=activation, 
                               kernel_initializer=init))
  model.add(keras.layers.Dropout(dropout))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add(keras.layers.Dense(128, activation=activation, 
                               kernel_initializer=init))
  model.add(keras.layers.Dropout(dropout))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add( keras.layers.Dense(128, activation=activation, 
                                kernel_initializer=init))
  model.add(keras.layers.Dropout(dropout))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  model.add(keras.layers.Dense(128, activation=activation, 
                               kernel_initializer=init))
  if batchNormalization:
    model.add(keras.layers.BatchNormalization()) 

  #NUEVA CAPA
  model.add(keras.layers.Dense(2, activation=activation, 
                               kernel_initializer=init))

  model.add(keras.layers.Dense(10, activation='softmax', 
                               kernel_initializer=init))

  return model

def plt_decision_boundary(model, features, labels):
  submodel= keras.Model(inputs=model.input, outputs=model.layers[-2].output)
  internal_data= submodel.predict(features)
  
  # Grid generation
  grid_len=1000
  xmin=np.amin(internal_data, axis=0)[0]
  xmax=np.amax(internal_data, axis=0)[0]

  ymin=np.amin(internal_data, axis=0)[1]
  ymax=np.amax(internal_data, axis=0)[1]
  Xgrid, Ygrid=np.meshgrid(np.linspace(xmin,xmax,grid_len), np.linspace(ymin,ymax,grid_len))

  # Grid prediction
  softMaxModel= keras.Model(inputs=model.layers[-1].input, 
                          outputs=model.layers[-1].output)
  
  testpoints = np.c_[Xgrid.ravel(), Ygrid.ravel()]
  predicted_data= np.argmax(softMaxModel.predict(testpoints), axis=1)
  
  #Plot Grid
  predicted_data= predicted_data.reshape(-1, grid_len)
  plt.contourf(Xgrid, Ygrid, predicted_data, 10, 
              cmap=plt.cm.gnuplot2, alpha=0.7)

  #Plot Samples
  featureX = np.array([internal_data[i][0] for i in range(len(internal_data))])
  featureY = np.array([internal_data[i][1] for i in range(len(internal_data))])

  for j in range(10):  
    plt.scatter(x=featureX[labels==j], y=featureY[labels==j], 
                alpha=0.5, color=plt.cm.gnuplot2(j/10), label=str(j))

  plt.legend()
  plt.show()