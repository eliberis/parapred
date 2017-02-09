from data_provider import load_data_matrices
from model import get_model
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

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
    print(labels_test.shape)
    print(test_probabilities.shape)
    prec, rec, thresholds = metrics.precision_recall_curve(
        labels_test.flatten(), test_probabilities.flatten())
    plt.plot(rec, prec)
    plt.plot(abip_rec, abip_pre)

    plt.ylabel("Precision")
    plt.yticks(np.linspace(0.3, 1, 8))

    plt.xlabel("Recall")
    plt.xticks(np.linspace(0.0, 1, 11))

    plt.savefig("proc.png")

def main():
    examples, labels, ags, params = load_data_matrices()
    print("Examples", examples.shape)
    print("Labels", labels.shape)
    print("Ags", ags.shape)

    model = get_model(params["max_ag_len"], params["max_cdr_len"])
    print(model.summary())

    test_size = round(len(examples) * 0.05)
    ags_train = ags[:-test_size]
    examples_train = examples[:-test_size]
    labels_train = labels[:-test_size]
    ags_test = ags[-test_size:]
    examples_test = examples[-test_size:]
    labels_test = labels[-test_size:]

    history = model.fit([ags_train, examples_train], labels_train,
                        batch_size=32, nb_epoch=30,
                        sample_weight=np.squeeze(labels_train * 5 + 1))

    model.save_weights("drop-5-30-noval-lstm64-allin.h5")

    plot_prec_rec_curve(model, ags_test, examples_test, labels_test)

if __name__ == "__main__":
    main()