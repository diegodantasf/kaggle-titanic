import argparse
import sys
sys.path.append('..')
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from preprocessor.naive import NaiveProcessor

np.random.seed(1337)

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
    parser.add_argument('--train', help='path to train.csv from Kaggle (Titanic)', type=str, default='../data/train.csv')
    parser.add_argument('--checkpoint-dir', help='path to output the trained model', type=str, default='../checkpoint/linear')
    parser.add_argument('--save', help='whether to save the model or not', action='store_true')
    args = parser.parse_args()
    
    train = pd.read_csv(args.train)

    processor = NaiveProcessor(train)
    X, y = processor(train, with_label=True)

    # Model
    keras.backend.set_floatx('float64')

    model = keras.models.Sequential([
        keras.layers.Dense(1, input_dim=8, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01))])
    
    model.compile(
        optimizer='Adam', 
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()])
    
    print(model.summary())
    
    callback_list = []
    
    if args.save:
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath= args.checkpoint_dir, 
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1)
        callback_list.append(checkpoint)

    h = model.fit(X, y, batch_size=32, validation_split=0.2, epochs=512, verbose=1,
        shuffle=True, workers=8, use_multiprocessing=True, callbacks=callback_list)
    
    plot_learning_curves(h)
#%%
if __name__ == '__main__':
    main()
