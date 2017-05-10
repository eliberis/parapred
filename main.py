from data_provider import *
from structure_processor import *
from evaluation import *
from model import *
from plotting import *
from keras.callbacks import LearningRateScheduler
import numpy as np

def single_run():
    train_set, test_set, params = open_dataset()
    # kfold_cv_eval(
    #     lambda: get_model(params["max_ag_len"], params["max_cdr_len"]),
    #     combine_datasets(train_set, test_set))
    # return

    max_ag_len = params["max_ag_len"]
    max_cdr_len = params["max_cdr_len"]
    pos_class_weight = params["pos_class_weight"]

    print("Max AG length:", max_ag_len)
    print("Max CDR length:", max_cdr_len)
    print("Pos class weight:", pos_class_weight)

    model = get_model(params["max_ag_len"], params["max_cdr_len"])
    print(model.summary())

    ags_train, cdrs_train, lbls_train, mask_train = train_set
    ags_test, cdrs_test, lbls_test, mask_test = test_set
    example_weight = np.squeeze((lbls_train * 1.5 + 1) * mask_train)

    rate_schedule = lambda e: 0.001 if e >= 5 else 0.01

    history = model.fit([ags_train, cdrs_train, np.squeeze(mask_train)],
                        lbls_train,
                        batch_size=32, epochs=17,
                        sample_weight=example_weight,
                        callbacks=[LearningRateScheduler(rate_schedule)])

    model.save_weights("abip-sets.h5")

    probs_test = model.predict([ags_test, cdrs_test, np.squeeze(mask_test)])

    test_seq_lens = np.sum(np.squeeze(mask_test), axis=1)
    probs_flat = flatten_with_lengths(probs_test, test_seq_lens)
    lbls_flat = flatten_with_lengths(lbls_test, test_seq_lens)

    pos_idx = probs_test > 0.5
    pred_test = np.zeros_like(probs_test)
    pred_test[pos_idx] = 1
    pred_flat = flatten_with_lengths(pred_test, test_seq_lens)

    print(confusion_matrix(lbls_flat, pred_flat))

    plot_roc_curve(lbls_flat, probs_flat)
    plot_prec_rec_curve([lbls_flat], [probs_flat],
                        output_filename="abip-sets.pdf")

    # plot_stats(history)
    # annotate_and_save_test_structures(probs_test)


def crossvalidation_eval():
    train_set, test_set, params = open_dataset()
    model_factory = \
        lambda: get_model(params["max_ag_len"], params["max_cdr_len"])
    dataset = combine_datasets(train_set, test_set)

    for i in range(10):
        print("Crossvalidation run", i+1)
        output_file = "data/seq_eval/run-{}.p".format(i)
        weights_template = "data/seq_eval/weights/run-" + str(i) + "-fold-{}.h5"
        kfold_cv_eval(model_factory, dataset, output_file, weights_template)


def process_cv_results():
    probs = []
    labels = []
    for r in range(10):
        result_filename = "runs/run-{}.p".format(r)
        with open(result_filename, "rb") as f:
            lbl_mat, prob_mat, mask_mat = pickle.load(f)

        seq_lens = np.sum(np.squeeze(mask_mat), axis=1)
        p = flatten_with_lengths(prob_mat, seq_lens)
        l = flatten_with_lengths(lbl_mat, seq_lens)

        probs.append(p)
        labels.append(l)

    plot_prec_rec_curve(labels, probs,
                        plot_name="PR curve for sequence-only model",
                        output_filename="seq-only.pdf")


if __name__ == "__main__":
    crossvalidation_eval()