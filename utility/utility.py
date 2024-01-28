# write a repeatable functions
import matplotlib.pyplot as plt
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


def plot_confusion_matrix(true_y, predicted_y):
    cm=confusion_matrix(true_y, predicted_y)
    display=ConfusionMatrixDisplay(cm)
    
    return display.plot()

def plot_ROC(true_y, predicted_y):
    fpr, tpr, _ = roc_curve(true_y.ravel(), predicted_y.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    return plt.show()


def plot_precision_recall_curve(true_y, predicted_y):
    precision, recall, _ = precision_recall_curve(true_y, predicted_y)

    # Calculate the area under the precision-recall curve
    average_precision = average_precision_score(true_y, predicted_y)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve (AUC = {:.2f})'.format(average_precision))
    return plt.show()

def model_performance_curves(model_history, tr_performance_para, val_performance_para, plot):
    """
    input:
        model_history: history of trained mdoel
        tr_performance_para: training accuracy or loss
        val_performance_para: validation accuracy or loss
        plot: label to y axis accuracy or loss
    
    """
    plt.plot(model_history.history[tr_performance_para])
    plt.plot(model_history.history[val_performance_para])
    plt.title(plot)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    return plt.show()

