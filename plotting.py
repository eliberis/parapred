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

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

    num_runs = len(labels_test)
    precs = np.zeros((num_runs, 10000))
    recalls = np.linspace(0.0, 1.0, num=10000)

    for i in range(num_runs):
        l = labels_test[i]
        p = probs_test[i]

        prec, rec, _ = metrics.precision_recall_curve(l.flatten(), p.flatten())

        # Maximum interpolation
        for j in range(len(prec)):
            prec[j] = prec[:(j+1)].max()

        prec = list(reversed(prec))
        rec = list(reversed(rec))

        for j, recall in enumerate(recalls):  # Inefficient, but good enough
            for p, r in zip(prec, rec):
                if r >= recall:
                    precs[i, j] = p
                    break

    avg_prec = np.average(precs, axis=0)
    err_prec = np.std(precs, axis=0)

    plt.plot(recalls, avg_prec, c="#0072CF", label="This method")

    btm_err = avg_prec-2*err_prec
    btm_err[btm_err < 0.0] = 0.0
    top_err = avg_prec+2*err_prec
    top_err[top_err > 1.0] = 1.0

    plt.fill_between(recalls, btm_err, top_err, facecolor="#68ACE5")

    abip_rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    abip_pre = \
        np.array([0.77, 0.74, 0.66, 0.61, 0.56, 0.51, 0.50, 0.48, 0.44, 0.415])
    abip_std = \
        np.array([0.06, 0.04, 0.031, 0.028, 0.026, 0.023, 0.02, 0.015, 0.013, 0.012])

    plt.scatter(abip_rec, abip_pre, c='#EA7125', s=10, label="Antibody i-Patch")
    plt.errorbar(abip_rec, abip_pre, yerr=2*abip_std, c="#EA7125")

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
