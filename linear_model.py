import argparse

import pandas as pd
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt

def preprocess(df):
    drop_names = ['Name', 'Ticket', 'PassengerId']
    to_categorical = ['Embarked', 'Cabin', 'Sex']

    df.drop(columns=drop_names, inplace=True)
    for s in to_categorical:
        df[s] = df[s].astype('category').cat.codes

def plot_learning_curves(h):
        losses, accuracies = h.history['loss'], h.history['binary_accuracy']
        val_losses, val_accuracies = h.history['val_loss'], h.history['val_binary_accuracy']
        
        plt.plot(losses, label='Train loss')
        plt.plot(val_losses, label='Validation loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(accuracies, label='Train accuracy')
        plt.plot(val_accuracies, label='Validation accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Welcome! Hope you know what you are doing.')
    parser.add_argument('--train', help='train.csv from Kaggle (Titanic)', type=str, required=True)
    parser.add_argument('--test', help='test.csv from Kaggle (Titanic)', type=str, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    
    preprocess(test)
    preprocess(train)

    X = train.loc[:, 'Pclass':].to_numpy()
    y = train.loc[:, 'Survived'].to_numpy().reshape((-1, 1))
    X_unknow = test.to_numpy()

    # Remove (bad) samples that do not contain all features
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]; y = y[mask]

    # Split dataset
    # 80% train 20% test
    X_train = X[:int(0.8 * X.shape[0])]; y_train = y[:int(0.8 * X.shape[0])]
    X_test = X[int(0.8 * X.shape[0]):]; y_test = y[int(0.8 * X.shape[0]):]

    keras.backend.set_floatx('float64')

    model = keras.models.Sequential([
    keras.layers.Dense(1, input_dim=8, activation='linear', kernel_regularizer=keras.regularizers.l2(0.01))])
    
    model.compile(
        optimizer='Adam', 
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    
    print(model.summary())
    
    h = model.fit(X_train, y_train, batch_size=32, validation_split=0.33, epochs=64, verbose=1,
        workers=8, use_multiprocessing=True
    )
    
    plot_learning_curves(h)
    
    loss, m = model.evaluate(X_test, y_test, verbose=0)
    
    print('Loss: %f' % loss)
    print('Accuracy %f' % m)

    # predictions = model.predict(X_unknow)


if __name__ == '__main__':
    main()