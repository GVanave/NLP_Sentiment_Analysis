# Sentiment Classification using LSTM and Bidirectional LSTM

## Overview

This project aims to perform sentiment classification using two different architectures: LSTM and Bidirectional LSTM. The sentiment classification is conducted on a dataset containing labeled reviews, with the objective of predicting whether a given review expresses positive or negative sentiment.

## Folders and Files

1. **model_training.ipynb**: This Jupyter notebook serves as the primary environment for training the sentiment classification models. It utilizes the `model.py` file for model development, `data.py` for data preprocessing, and `utilities.py` for model evaluation.

2. **models/model.py**: This Python script contains the implementation of the LSTM and Bidirectional LSTM models. It defines the architecture, layers, and configurations for training the sentiment classification models.

3. **data/data.py**: This Python script includes basic functions for preprocessing the dataset. It handles tasks such as loading the data, cleaning, and preparing it for training.

4. **utilities/utilities.py**: This Python script provides utility functions used for evaluating model performance. It includes functions for generating accuracy and loss curves, confusion matrices, and other evaluation metrics.

5. **model_evaluation**: contains the curves about the model performance.
        model_evaluation/accuracy_curve_bidir_LSTM_40epoch.png
        model_evaluation/accuracy_curve_LSTM_40epoch.png
        model_evaluation/Bidir_LSTM_confusionmetric_40epoch.png
        model_evaluation/Classification_report_Bidir_LSTM_40epoch.png
        model_evaluation/Classification_report_LSTM_40epoch.png
        model_evaluation/loss_curve_Bidir_LSTM_40epoch.png
        model_evaluation/loss_curve_LSTM_40epoch.png
        model_evaluation/model_evaluationLSTM_confusionmetric_40epoch.png

## Libraries Used

- pandas==2.2.0
- nltk==3.8.1
- numpy==1.26.3
- tensorflow==2.15.0
- matplotlib==3.8.2
- scikit-learn==1.4.0

## Instructions

1. **Environment Setup**: Ensure you have the required libraries installed. You can install them using the following command:

   ```bash
   pip install -r requirements.txt


## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)

