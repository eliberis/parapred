from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def plot_stats(history, plot_filename="stats.pdf"):
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.title('Metrics vs number of epochs')

    plt.subplot(3, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(history.history['false_pos'])
    plt.plot(history.history['val_false_pos'])
    plt.ylabel('False positive rate')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(history.history['false_neg'])
    plt.plot(history.history['val_false_neg'])
    plt.ylabel('False negative rate')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')

    plt.savefig(plot_filename)


def plot_prec_rec_curve(labels_test, probs_test, plot_name="",
                        output_filename="proc.pdf"):
    plt.figure(figsize=(4.5, 3.5))

    prec, rec, thresholds = metrics.precision_recall_curve(
        labels_test.flatten(), probs_test.flatten())

    # Maximum interpolation
    for i in range(len(prec)):
        prec[i] = prec[:(i+1)].max()

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

    plt.plot(rec, prec, c="#0072CF", label="ParaPred")

    plt.ylabel("Precision")
    plt.xlabel("Recall")

    plt.title(plot_name)
    plt.legend()
    plt.savefig(output_filename)


def plot_roc_curve(labels_test, probs_test, plot_name="ROC Curve",
                   output_filename="roc.pdf"):
    plt.figure(figsize=(3.5, 3.5))

    fpr, tpr, thresholds = \
        metrics.roc_curve(labels_test.flatten(), probs_test.flatten())

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

    plt.plot(fpr, tpr, c="#0072CF", label="ParaPred")

    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")

    plt.title(plot_name)
    plt.savefig(output_filename)


def confusion_matrix(true, pred):
    return metrics.confusion_matrix(true, pred)