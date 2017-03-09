from data_provider import open_dataset
from model import get_model
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracies(history):
    plt.interactive(False)

    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("prec.png")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")

def plot_prec_rec_curve(model, ags_test, examples_test, labels_test):
    plt.interactive(False)

    abip_rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    abip_pre = \
        np.array([0.78, 0.74, 0.66, 0.62, 0.56, 0.51, 0.5, 0.48, 0.45, 0.44])

    test_probabilities = model.predict([ags_test, examples_test])
    prec, rec, thresholds = metrics.precision_recall_curve(
        labels_test.flatten(), test_probabilities.flatten())

    # Maximum interpolation
    for i in range(len(prec)):
        prec[i] = prec[:(i+1)].max()

    plt.plot(rec, prec)
    plt.plot(abip_rec, abip_pre)

    plt.ylabel("Precision")
    plt.yticks(np.linspace(0.3, 1, 8))

    plt.xlabel("Recall")
    plt.xticks(np.linspace(0.0, 1, 11))

    plt.savefig("proc.png")

def main():
    ags, cdrs, lbls, params = open_dataset()

    print("Ags:", ags.shape)
    print("CDRs:", cdrs.shape)
    print("Labels:", lbls.shape)

    max_ag_len = params["max_ag_len"]
    max_cdr_len = params["max_cdr_len"]

    print("Max AG length:", max_ag_len)
    print("Max CDR length:", max_cdr_len)

    model = get_model(params["max_ag_len"], params["max_cdr_len"])
    print(model.summary())

    np.random.seed(seed=0)  # TODO replace with stratified split
    test_size = round(len(cdrs) * 0.20)
    indices = np.random.permutation(len(cdrs))

    ags_train = ags[indices[:-test_size]]
    cdrs_train = cdrs[indices[:-test_size]]
    lbls_train = lbls[indices[:-test_size]]
    ags_test = ags[indices[-test_size:]]
    cdrs_test = cdrs[indices[-test_size:]]
    lbls_test = lbls[indices[-test_size:]]
    example_weight = np.squeeze(lbls_train * 5 + 1)  # 6-to-1 in favour of 1

    history = model.fit([ags_train, cdrs_train], lbls_train,
                        batch_size=32, nb_epoch=30,
                        sample_weight=example_weight)

    model.save_weights("current.h5")
    plot_prec_rec_curve(model, ags_test, cdrs_test, lbls_test)

if __name__ == "__main__":
    main()