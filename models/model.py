import tensorflow 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, LSTM, Dropout, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


custom_optimizer = Adam(learning_rate=0.01)


# define the model
def LSTM_model(voc_length, embedding_max, max_len):
    custom_optimizer = Adam(learning_rate=0.005)
    
    model = Sequential()
    embedding_layer = Embedding(voc_length, 50, weights = [embedding_max], input_length = max_len, trainable = False)
    model.add(embedding_layer)
    model.add(LSTM(60))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = "sigmoid"))
    

    model.compile(loss="binary_crossentropy", optimizer=custom_optimizer, metrics=["accuracy"])
    return model



def LSTM_model_not_pre(voc_length, max_len):
    custom_optimizer = Adam(learning_rate=0.01)
    
    model = Sequential()
    embedding_layer = Embedding(voc_length, 200, input_length = max_len)
    model.add(embedding_layer)
    model.add(LSTM(60))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=custom_optimizer, metrics=["accuracy"])
    return model
def RNN_model():
    pass

def cov_classification(voc_length, embedding_max, max_len):
    model = Sequential()
    embedding_layer = Embedding(voc_length, 50, weights = [embedding_max], input_length = max_len, trainable = False)
    model.add(embedding_layer)
    model.add(Conv1D(30, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3), bias_regularizer=regularizers.l2(2e-3)))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(10, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3), bias_regularizer=regularizers.l2(2e-3)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    return model