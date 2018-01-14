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


def plot_pr_curve(labels_test, probs_test, colours=("#0072CF", "#68ACE5"),
                  label="This method", plot_fig=None):
    if plot_fig is None:
        plot_fig = plt.figure(figsize=(4.5, 3.5), dpi=300)
    ax = plot_fig.gca()

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

    ax.plot(recalls, avg_prec, c=colours[0], label=label)

    btm_err = avg_prec - 2 * err_prec
    btm_err[btm_err < 0.0] = 0.0
    top_err = avg_prec + 2 * err_prec
    top_err[top_err > 1.0] = 1.0

    ax.fill_between(recalls, btm_err, top_err, facecolor=colours[1])

    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.legend()

    return plot_fig


def plot_abip_pr(plot_fig=None):
    if plot_fig is None:
        plot_fig = plt.figure(figsize=(4.5, 3.5), dpi=300)
    ax = plot_fig.gca()

    abip_rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    abip_pre = np.array([0.77, 0.74, 0.66, 0.61, 0.56,
                         0.51, 0.50, 0.48, 0.44, 0.415])
    abip_std = np.array([0.06, 0.04, 0.031, 0.028, 0.026,
                         0.023, 0.02, 0.015, 0.013, 0.012])

    ax.errorbar(abip_rec, abip_pre, yerr=2 * abip_std, label="Antibody i-Patch",
                fmt='o', mfc="#EA7125", mec="#EA7125", ms=3,
                ecolor="#F3BD48", elinewidth=1, capsize=3)

    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.legend()

    return plot_fig


def plot_roc_curve(labels_test, probs_test, colours=("#0072CF", "#68ACE5"),
                   label="This method", plot_fig=None):
    if plot_fig is None:
        plot_fig = plt.figure(figsize=(3.7, 3.7), dpi=400)
    ax = plot_fig.gca()

    num_runs = len(labels_test)
    tprs = np.zeros((num_runs, 10000))
    fprs = np.linspace(0.0, 1.0, num=10000)

    for i in range(num_runs):
        l = labels_test[i]
        p = probs_test[i]

        fpr, tpr, _ = metrics.roc_curve(l.flatten(), p.flatten())

        for j, fpr_val in enumerate(fprs):  # Inefficient, but good enough
            for t, f in zip(tpr, fpr):
                if f >= fpr_val:
                    tprs[i, j] = t
                    break

    avg_tpr = np.average(tprs, axis=0)
    err_tpr = np.std(tprs, axis=0)

    ax.plot(fprs, avg_tpr, c=colours[0], label=label)

    btm_err = avg_tpr - 2 * err_tpr
    btm_err[btm_err < 0.0] = 0.0
    top_err = avg_tpr + 2 * err_tpr
    top_err[top_err > 1.0] = 1.0

    ax.fill_between(fprs, btm_err, top_err, facecolor=colours[1])

    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")

    ax.legend()

    return plot_fig


def plot_binding_profiles(contact, parapred, colours=("#0072CF", "#D6083B"),
                          save_as="binding_profiles.eps"):
    plt.rcParams["font.family"] = "Arial"

    plot_fig = plt.figure(figsize=(4.8, 3.7), dpi=400)
    ax = plot_fig.gca()

    # Is order in contact and parapred values always the same?
    ind = np.arange(len(contact.keys()))
    width = 0.35

    ax.bar(ind, np.array(list(contact.values())), width,
            color=colours[0], label='Contact')
    ax.bar(ind + width, np.array(list(parapred.values())), width,
            color=colours[1], label='Model\'s predictions')

    ax.set_ylabel('Relative binding frequency')
    ax.set_title('Residue type binding profile')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(contact.keys())

    plt.legend()
    plt.savefig(save_as)
