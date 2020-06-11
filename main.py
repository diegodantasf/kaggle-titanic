import argparse
import pandas as pd
import numpy as np

from tensorflow import keras
from preprocessor.naive import NaiveProcessor

def main():
    parser = argparse.ArgumentParser(description=
    """
        Hi.
    """)
    parser.add_argument('--train', help='path to train.csv from Kaggle (Titanic)', type=str, default='data/train.csv')
    parser.add_argument('--test', help='path to test.csv from Kaggle (Titanic)', type=str, default='data/test.csv')
    parser.add_argument('--checkpoint-dir', help='path to directory of the model', type=str, default='checkpoint/linear')
    parser.add_argument('-p', help='type of preprocessing (available: naive)', type=str, default='naive')
    parser.add_argument('-o', help='name of output file with results to submit on Kaggle', type=str, default='result')
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    model = keras.models.load_model(args.checkpoint_dir)    
    if args.p:
        processor = NaiveProcessor(train)
    
    X, y = processor(test)

    predictions = model.predict_classes(X) # Threshold is 0.5
    
    predictions = pd.DataFrame(data=predictions, columns=['Survived'], dtype=int)
    
    result = pd.DataFrame(test['PassengerId']).join(predictions)

    print (result.head(10))

    result.to_csv(args.o + '.csv', index=False)

if __name__ == '__main__':
    main()
