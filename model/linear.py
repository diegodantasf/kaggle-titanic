import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


def preprocess(df):
    drop_names = ['Name', 'Ticket', 'PassengerId']
    to_categorical = ['Embarked', 'Cabin', 'Sex']

    df.drop(columns=drop_names, inplace=True)
    for s in to_categorical:
        df[s] = df[s].astype('category').cat.codes

    X = df.loc[:, 'Pclass':].to_numpy()
    y = df.loc[:, 'Survived'].to_numpy().reshape((-1, 1))

    # Remove (bad) samples that do not contain all features
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]; y = y[mask]

    # Shuffle

    np.random.seed(1337) # Fixed seed

    permutation = np.arange(X.shape[0])
    np.random.shuffle(permutation)
    X = X[permutation]; y = y[permutation]

    # Split dataset
    # 80% train 20% test
    X_train, y_train = X[:int(0.8 * X.shape[0])], y[:int(0.8 * X.shape[0])]
    X_test, y_test = X[int(0.8 * X.shape[0]):], y[int(0.8 * X.shape[0]):]

    # Normalize
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_X = scaler_X.fit(X_train)
    scaler_y = scaler_y.fit(y_train)
    
    X_train, y_train = scaler_X.transform(X_train), scaler_y.transform(y_train)
    X_test, y_test = scaler_X.transform(X_test), scaler_y.transform(y_test)

    return X_train, y_train, X_test, y_test

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
    parser = argparse.ArgumentParser(description=
    """
        Welcome! Hope you know what you are doing.
        Run this code from the model directory if you want to use the default values for arguments.
    """)
    parser.add_argument('--train', help='path totrain.csv from Kaggle (Titanic)', type=str, default='../data/train.csv')
    parser.add_argument('--save', help='whether to save the model or not (0 or 1)', type=bool, default=False)
    parser.add_argument('--checkpoint-dir', help='path to output the trained model', type=str, default='../checkpoint/linear')
    args = parser.parse_args()

    train = pd.read_csv(args.train)

    X_train, y_train, X_test, y_test = preprocess(train)

    # Model

    keras.backend.set_floatx('float64')

    model = keras.models.Sequential([
        keras.layers.Dense(1, input_dim=8, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01))])
    
    model.compile(
        optimizer='Adam', 
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()])
    
    print(model.summary())


    if args.save:
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath= args.checkpoint_dir, 
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        h = model.fit(X_train, y_train, batch_size=32, validation_split=0.33, epochs=512, verbose=1,
            workers=8, use_multiprocessing=True, callbacks=[checkpoint])
    else:
        h = model.fit(X_train, y_train, batch_size=32, validation_split=0.33, epochs=512, verbose=1,
            workers=8, use_multiprocessing=True)
    
    plot_learning_curves(h)
    
    loss, m = model.evaluate(X_test, y_test, verbose=0)
    
    print('Train loss: %f' % loss)
    print('Train accuracy %f' % m)

if __name__ == '__main__':
    main()
