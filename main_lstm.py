from utils.preprocessor import Preprocessor
from utils.preprocessor_lstm import PreprocessorLSTM
import sys

import matplotlib.pyplot  as plt

from sklearn.metrics import classification_report
from keras.layers import Embedding, LSTM, Dropout, Dense, Input, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np


if __name__ =="__main__":
    dataset_binary = "datasets/binary.csv"
    dataset_ite = "datasets/Dataset Twitter Fix - Indonesian Sentiment Twitter Dataset Labeled (1).csv"
    save_dir = "datasets/merged_dataset.pkl"
    pre = PreprocessorLSTM(save_dir, dataset_binary, dataset_ite)
    
    vocab_size, weight, x_train, x_val, y_train, y_val = pre.preprocessing_glove()
    # x_train, x_val, y_train, y_val, vocab_size, embedding_size, weight = pre.main()
    print("x_t = ", x_train.shape,"x_v = ", x_val.shape)
    print("y_t = ", y_train.shape,"y_v = ", y_val.shape)
    input_size = x_train.shape
    # input = Input(shape = input_size)
    # embedding = Embedding(input)
    embedding_layer = Embedding(input_dim=vocab_size, output_dim = 50, weights=[weight], trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(100))

    model.add(Dropout(0.1))
    model.add(Dense(12, activation='sigmoid'))

    model.summary()

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=10, batch_size=256, verbose=1)
    plt.plot(train.history['accuracy'], label='accuracy') 
    plt.plot(train.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()