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


def plot_prec_rec_curve(labels_test, probs_test, plot_name="",
                        output_filename="proc.png"):
    plt.figure(figsize=(4.5, 3.5))

    abip_rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    abip_pre = \
        np.array([0.77, 0.74, 0.66, 0.61, 0.56, 0.51, 0.50, 0.48, 0.44, 0.415])

    prec, rec, thresholds = metrics.precision_recall_curve(
        labels_test.flatten(), probs_test.flatten())

    # Maximum interpolation
    for i in range(len(prec)):
        prec[i] = prec[:(i+1)].max()

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

    plt.plot(rec, prec, c="#0072CF", label="ParaPred")
    plt.scatter(abip_rec, abip_pre, c='#EA7125', label="Antibody i-Patch")

    plt.ylabel("Precision")
    plt.yticks(np.linspace(0.2, 1, 9))

    plt.xlabel("Recall")
    plt.xticks(np.linspace(0.0, 1, 11))

    plt.title(plot_name)
    plt.legend()
    plt.savefig(output_filename)
