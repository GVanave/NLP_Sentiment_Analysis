import pandas as pd 
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



def data_cleaning(train_df):
    
    """
    input
        train_df: df

    output
        train_df: cleaned review column
        
    """
    ps=PorterStemmer()
    
    stopword=list(stopwords.words("english"))
    # pattern = r'\[.*?\]'
    # train_df["review"] = train_df["review"].apply(lambda x: re.sub(pattern, '', x))
    # train_df["review"] = train_df["review"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    training_dataset=[]
  
    for i in range(len(train_df)):
        review=re.sub("[^a-zA-Z]"," ",train_df["review"][i])
        review=review.lower()
        review=review.split()
        review=[ps.stem(word) for word in review if not word  in stopword]
        review=" ".join(review)
        training_dataset.append(review)
  
    return training_dataset


def data_preprocessing(train_data_path, test_data_path):
    """
    inputs: train_data_path and test_data_path

    outputs: preprocessed data frames 
    """

    # read the train and test data
    train_df = pd.read_csv(train_data_path, delimiter = "\t")
    test_df = pd.read_csv(test_data_path, delimiter = "\t")

    train_df[["movie_id", "rating"]] = train_df["id"].apply(lambda x : pd.Series(x.split('_')))
    test_df[["movie_id", "rating"]] = test_df["id"].apply(lambda x : pd.Series(x.split('_')))

    test_df["is_common_id"] = test_df["movie_id"] == train_df["movie_id"]
    df_test_mofified = test_df.drop(test_df[test_df["is_common_id"] == True].index, axis = 0)

    corpus = data_cleaning(train_df)

    X = corpus
    y = train_df["sentiment"]

    return X, y, df_test_mofified