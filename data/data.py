import pandas as pd 
import re
from nltk.corpus import stopwords
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def check_null(df1, df2):
    null_columns_df1 = df1.columns[df1.isnull().any()].tolist()
    null_columns_df2 = df2.columns[df2.isnull().any()].tolist()

    if null_columns_df1 or null_columns_df2:
        print("There are null values in the DataFrames.")
        if null_columns_df1:
            print("Columns with null values in df1:", null_columns_df1)
        if null_columns_df2:
            print("Columns with null values in df2:", null_columns_df2)
    else:
        print("No null values in the DataFrames.")
    return None

def data_cleaning(df):
    
    """
    input
        train_df: df

    output
        corpus: well cleaned data
        
    """
    # pattern = r'\[.*?\]'
    # df["review"] = df["review"].apply(lambda x: re.sub(pattern, '', x))
    # df["review"] = df["review"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    # ps = PorterStemmer()
    stopword = list(stopwords.words("english"))
    
    corpus=[]
  
    for i in range(len(df)):
        review=re.sub("[^a-zA-Z\s]"," ",df["review"][i])
        review=review.lower()
        review=review.split()
        review=[word for word in review if not word  in stopword]
        review=" ".join(review)
        corpus.append(review)
  
    return corpus


def remove_col(df):

    """
    input:
        df: input dataframe, train or test data
    output:
        df: modified dataframe, remove movies with neural sentiment, dropped id column
    """

    # 1:positive 0:negative 2:neutral
    df.loc[df["rating"] > 7, "Rating"] = 1
    df.loc[df["rating"] < 5, "Rating"] = 0
    df.loc[(df["rating"] >= 5) & (df["rating"] <= 7), "Rating"] = 2
    df["Rating"] = df["Rating"].astype(int)
    df = df[df["Rating"] != 2]
    df = df.drop("id", axis=1)
    df = df.copy()

    return df


def change_datatype(train_df, test_df):

    """
    input:
        df: train or test dataframe
    output:
        df: dataframe with mofidied data types 
    """
    train_df["movie_id"] = train_df["movie_id"].astype(int)
    train_df["rating"] = train_df["rating"].astype(int)

    test_df["movie_id"] = test_df["movie_id"].astype(int)
    test_df["rating"] = test_df["rating"].astype(int)

    return train_df, test_df

def extract_features(train_df, test_df):

    # feature engineering on id, seperate movie id and rating
    train_df[["movie_id", "rating"]] = train_df["id"].apply(lambda x : pd.Series(x.split('_')))
    test_df[["movie_id", "rating"]] = test_df["id"].apply(lambda x : pd.Series(x.split('_')))

    return train_df, test_df
    
def data_preprocessing(train_data_path, test_data_path):
    """
    inputs: 
        train_data_path and test_data_path
    outputs: 
        train_corpus: cleaned review col from train_df
        test_corpus: cleaned review col from test_df
        y_train: labels for training data
    """

    # read the train and test data
    train_df = pd.read_csv(train_data_path, delimiter = "\t")
    test_df = pd.read_csv(test_data_path, delimiter = "\t")

    # check null values
    check_null(train_df, test_df)

    # cleaning the reviews, removing unwanted information
    train_df["review_cleaned"] = data_cleaning(train_df)
    test_df["review_cleaned"] = data_cleaning(test_df)

    train_df, test_df = extract_features(train_df, test_df)

    # modify the data type of newly created columns for further analysis
    train_df, test_df = change_datatype(train_df, test_df)

    # test data must be uniqe so we remove common entried of movie_id
    test_df_modified = test_df[test_df["movie_id"] != train_df["movie_id"]]
    
    # Modify dataframes by ignore natural reviews
    train_data = remove_col(train_df)
    test_data = remove_col(test_df_modified)

    print(train_data.shape)
    train_corpus = list(train_data["review_cleaned"])
    test_corpus = list(test_data["review_cleaned"])

    y_train = np.array((train_data["sentiment"]))
    y_test = np.array(test_data["Rating"])


    # return train_data, y, test_data
    return train_corpus, test_corpus, y_train, y_test

def apply_tokenizer(corpus):

    """
    input:
        corpus: well cleaned data
    output:
        tokenizered_corpus: corpus converted to tokens
        voc_length: number of unique words in the corpus
        tokenizer: object of tokenzier
    """
    # define the maximum size of the embedding vector, used fpr padding to have same diamention for all vector
    max_len = 200
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus = tokenizer.texts_to_sequences(corpus)
    tokenizered_corpus = pad_sequences(corpus, padding = 'post', maxlen = max_len)
    voc_length = len(tokenizer.word_index) + 1

    return tokenizered_corpus, voc_length, tokenizer

def load_pretrained_embedding(path_to_pretrained_embedding, corpus):

    """
    input:
        path_to_pretrained_embedding: we are using the pretrained weights for word2vec (glov)
        corpus: well cleaned data
    output:
        embedding_matrix: getting pretrained weights of embedding 
        tokenizered_corpus: corpus ready for training
    """

    tokenizered_corpus, voc_length, tokenizer = apply_tokenizer(corpus)
    embedding_dictonary = dict()
    glove_file = open(path_to_pretrained_embedding, encoding= "utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimentions = np.asarray(records[1:], dtype="float32")
        embedding_dictonary[word] = vector_dimentions
    glove_file.close()


    embedding_matrix = np.zeros((voc_length, 50))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embedding_dictonary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix, tokenizered_corpus, voc_length


if __name__ == "__main__":

    train_df_path = './labeledTrainData.tsv'
    test_df_path = "./testData.tsv"

    train_corpus, test_corpus, y_train = data_preprocessing(train_df_path, test_df_path)