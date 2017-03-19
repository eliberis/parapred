from data_provider import open_dataset
from model import get_model
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_fold as td
import tensorflow as tf
import random

SAVE_NAME = "mdl/paratope-pred"

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

def plot_prec_rec_curve(labels, predictions):
    plt.interactive(False)

    abip_rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    abip_pre = \
        np.array([0.78, 0.74, 0.66, 0.62, 0.56, 0.51, 0.5, 0.48, 0.45, 0.44])

    prec, rec, thresholds = metrics.precision_recall_curve(
        np.array(labels), np.array(predictions))

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
    dataset = open_dataset()

    # Shuffle and split into training and test sets
    random.seed(0)
    random.shuffle(dataset)
    test_size = round(len(dataset) * 0.20)
    train_set = dataset[:-test_size]
    test_set = dataset[-test_size:]

    train_and_save(train_set)
    lbls, preds = eval(test_set)
    plot_prec_rec_curve(lbls, preds)


def train_and_save(dataset, batch_size=32, num_epochs=30):
    model = get_model(keep_prob=0.85)
    compiler = td.Compiler.create(model)

    loss = compiler.metric_tensors['loss']
    train = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_set = compiler.build_loom_inputs(dataset)
    train_feed_dict = {}  # add feeds for e.g. dropout here

    for epoch, shuffled in enumerate(td.epochs(train_set, num_epochs), 1):
        ex_processed = 0
        for batch in td.group_by_batches(shuffled, batch_size):
            train_feed_dict[compiler.loom_input_tensor] = batch
            _, batch_loss = sess.run([train, loss], train_feed_dict)
            last_loss = sum(batch_loss)
            ex_processed += len(batch_loss)
            print('Epoch {0}/{1}: '
                  '{2}/{3} example(s) processed, loss {4:.3f}.'
                  .format(epoch, num_epochs, ex_processed, len(dataset),
                          last_loss))
        print('\n')

    saver = tf.train.Saver()
    saver.save(sess, SAVE_NAME)
    sess.close()


def eval(dataset, batch_size=32):
    model = get_model(keep_prob=1.00)
    compiler = td.Compiler.create(model)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "./" + SAVE_NAME)

    test_set = compiler.build_loom_inputs(dataset)
    feed_dict = {}

    predictions = []
    probs_metric = compiler.metric_tensors['probs']
    for batch in td.group_by_batches(test_set, batch_size):
        feed_dict[compiler.loom_input_tensor] = batch
        predictions.extend(sess.run(probs_metric, feed_dict))

    labels_flat = [lb for entry in dataset for lb in entry['lb']]
    return labels_flat, predictions


if __name__ == "__main__":
    main()
