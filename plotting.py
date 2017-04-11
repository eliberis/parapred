from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracies(history,
                    prec_filename="prec.png",
                    loss_filename="loss.png"):
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(prec_filename)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_filename)


def plot_prec_rec_curve(labels, predictions,
                        output_filename="proc.png"):
    plt.figure()

    abip_rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    abip_pre = \
        np.array([0.78, 0.74, 0.66, 0.62, 0.56, 0.51, 0.5, 0.48, 0.45, 0.44])

    prec, rec, thresholds = metrics.precision_recall_curve(
        labels.flatten(), predictions.flatten())

    # Maximum interpolation
    for i in range(len(prec)):
        prec[i] = prec[:(i+1)].max()

    plt.plot(rec, prec)
    plt.plot(abip_rec, abip_pre)

    plt.ylabel("Precision")
    plt.yticks(np.linspace(0.3, 1, 8))

    plt.xlabel("Recall")
    plt.xticks(np.linspace(0.0, 1, 11))

    plt.title(output_filename)
    plt.savefig(output_filename)
