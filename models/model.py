import tensorflow 
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


custom_optimizer = Adam(learning_rate=0.01)

# define the models
def LSTM_model(voc_length, embedding_max, max_len, dropout_p, hidden_neurons, lstm_neurons):
    
    model = Sequential()
    embedding_layer = Embedding(voc_length, max_len, weights = [embedding_max], trainable = False)
    model.add(embedding_layer)
    model.add(LSTM(lstm_neurons, dropout = dropout_p ,kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=custom_optimizer, metrics=["accuracy"])
    return model

def BidirLSTM_model(voc_length, embedding_max, max_len, dropout_p, hidden_neurons, lstm_neurons):
    model = Sequential()
    model.add(Embedding(voc_length, max_len , weights = [embedding_max], trainable = False))
    model.add(Bidirectional(LSTM(lstm_neurons)))
    model.add(Dense(1,activation='sigmoid')) # for binary classification
    model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model