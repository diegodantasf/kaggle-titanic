import argparse

import pandas as pd
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt

#%%
def main():
    try: 
        parser = argparse.ArgumentParser(description='Welcome! Hope you know what you are doing.')
        parser.add_argument('--train', help='train.csv from Kaggle (Titanic)', type=str, required=True)
        parser.add_argument('--test', help='test.csv from Kaggle (Titanic)', type=str, required=True)
        args = parser.parse_args()
        #print(args.train)
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.test)
    except:
        train_path = r'C:\Users\joyce\Desktop\Jeremie\github\kaggle-titanic\data\train.csv'
        test_path = r'C:\Users\joyce\Desktop\Jeremie\github\kaggle-titanic\data\test.csv'
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(train_path)
    
    try:
        print(os.getcwd())
        os.chdir(r'C:\Users\joyce\Desktop\Jeremie\github\kaggle-titanic')
    except:
        pass

    def preprocess(df):
        drop_names = ['Name', 'Ticket', 'PassengerId']
        to_categorical = ['Embarked', 'Cabin', 'Sex']

        df.drop(columns=drop_names, inplace=True)
        for s in to_categorical:
            df[s] = df[s].astype('category').cat.codes
    
    def plot_learning_curves(history):
        losses, accuracies = history.history['loss'], history.history['binary_accuracy']
        val_losses, val_accuracies = history.history['val_loss'], history.history['val_binary_accuracy']
        
        plt.plot(losses, label='Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(accuracies, label='Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show()
    
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
    
    history = model.fit(X_train, y_train, batch_size= 32, validation_split=0.33, epochs=1024, verbose=1)
    
    plot_learning_curves(history)
    
    loss, m = model.evaluate(X_test, y_test, verbose=0)
    
    print('Loss: %f' % loss)
    print('Accuracy %f' % m)

    # predictions = model.predict(X_unknow)



if __name__ == '__main__':
    main()

#%%