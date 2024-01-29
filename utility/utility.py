
import matplotlib.pyplot as plt
import datetime, os
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def data_analysis(df):
    return df.isna().sum()

def get_classification_report(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred)
    return accuracy, cls_report



def plot_data_balance(df):
    df["sentiment"].value_counts().plot(kind="bar")
    plt.xlabel("1:positive reviews" + " 0:negative reviews" )
    plt.ylabel("Counts/Frequncy")
    return plt.show()

def plot_confusion_matrix(true_y, predicted_y, model_type):
    cm=confusion_matrix(true_y, predicted_y)
    display=ConfusionMatrixDisplay(cm)
    print(f"confusion matrix for {model_type}")
    return display.plot()

def plot_ROC(true_y, predicted_y, model_type):
    fpr, tpr, _ = roc_curve(true_y.ravel(), predicted_y.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for model with: {model_type}')
    plt.legend(loc='lower right')
    return plt.show()


def plot_precision_recall_curve(true_y, predicted_y, model_type):
    precision, recall, _ = precision_recall_curve(true_y, predicted_y)

    # Calculate the area under the precision-recall curve
    average_precision = average_precision_score(true_y, predicted_y)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve (AUC = {average_precision}) for model with {model_type}')
    return plt.show()

def model_performance_curves(model_history, performance_para ,tr_performance_para, val_performance_para, model_type, path):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    """
    input:
        model_history: history of trained model
        performance_para: loss or accuracy
        tr_performance_para: training accuracy or loss
        val_performance_para: validation accuracy or loss
        plot: label to y axis accuracy or loss
    
    """
    plt.plot(model_history.history[tr_performance_para])
    plt.plot(model_history.history[val_performance_para])
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel(performance_para)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.title(f'{performance_para} curve of model with {model_type}')
    plt.savefig(rf"{path}\{model_type}_{performance_para}_{time}.png", format = "pdf")
    return plt.show()

