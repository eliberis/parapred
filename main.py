from data_provider import *
from evaluation import *
from model import *
from plotting import *
from keras.callbacks import LearningRateScheduler, EarlyStopping
import numpy as np


def single_run():
    dataset = open_dataset("data/sabdab_27_jun_95_90.csv")

    max_cdr_len = dataset["max_cdr_len"]
    pos_class_weight = dataset["pos_class_weight"]
    size = len(dataset["cdrs"])

    print("Max CDR length:", max_cdr_len)
    print("Pos class weight:", pos_class_weight)
    print("Number of structures:", size)

    model = ab_seq_model(max_cdr_len)
    print(model.summary())

    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]

    np.random.seed(0)  # For reproducibility
    indices = np.random.permutation(size)
    test_size = size // 10

    cdrs_train = cdrs[indices[:-test_size]]
    lbls_train = lbls[indices[:-test_size]]
    masks_train = masks[indices[:-test_size]]

    cdrs_test = cdrs[indices[-test_size:]]
    lbls_test = lbls[indices[-test_size:]]
    masks_test = masks[indices[-test_size:]]

    example_weight = np.squeeze((lbls_train * 1.5 + 1) * masks_train)
    test_ex_weight = np.squeeze((lbls_test * 1.5 + 1) * masks_test)

    rate_schedule = lambda e: 0.001 if e >= 10 else 0.01

    history = model.fit([cdrs_train, np.squeeze(masks_train)],
                        lbls_train, batch_size=32, epochs=150,
                        # Just a trial, not actual evaluation.
                        validation_data=([cdrs_test, np.squeeze(masks_test)],
                                         lbls_test, test_ex_weight),
                        sample_weight=example_weight,
                        callbacks=[LearningRateScheduler(rate_schedule),
                                   EarlyStopping(verbose=1, patience=3)])

    model.save_weights("sabdab.h5")

    probs_test = model.predict([cdrs_test, np.squeeze(masks_test)])

    test_seq_lens = np.sum(np.squeeze(masks_test), axis=1)
    probs_flat = flatten_with_lengths(probs_test, test_seq_lens)
    lbls_flat = flatten_with_lengths(lbls_test, test_seq_lens)

    compute_classifier_metrics([lbls_flat], [probs_flat], threshold=0.5)

    plot_roc_curve(lbls_flat, probs_flat)
    plot_prec_rec_curve([lbls_flat], [probs_flat],
                        output_filename="sabdab.pdf")

    # plot_stats(history)
    # annotate_and_save_test_structures(probs_test)


def crossvalidation_eval():
    dataset = open_dataset("data/sabdab_27_jun_95_90.csv")
    model_factory = \
        lambda: ab_seq_model(dataset["max_cdr_len"])

    for i in range(10):
        print("Crossvalidation run", i+1)
        output_file = "cv-ab-seq/run-{}.p".format(i)
        weights_template = "cv-ab-seq/weights/run-" + str(i) + "-fold-{}.h5"
        kfold_cv_eval(model_factory, dataset,
                      output_file, weights_template, seed=i)


def process_cv_results():
    probs = []
    labels = []
    for r in range(8):
        result_filename = "cv-ab-seq/run-{}.p".format(r)
        with open(result_filename, "rb") as f:
            lbl_mat, prob_mat, mask_mat = pickle.load(f)

        seq_lens = np.sum(np.squeeze(mask_mat), axis=1)
        p = flatten_with_lengths(prob_mat, seq_lens)
        l = flatten_with_lengths(lbl_mat, seq_lens)

        plot_roc_curve(l, p, plot_name="ROC", output_filename="roc.pdf")

        probs.append(p)
        labels.append(l)

    plot_prec_rec_curve(labels, probs,
                        plot_name="PR curve for the antibody sequence-only model",
                        output_filename="seq-only.pdf")

    compute_classifier_metrics(labels, probs, threshold=0.5)


# def patchdock_prepare():
#     _, test_set, params = open_dataset()
#     model = ab_seq_model(params["max_cdr_len"])
#     model.load_weights("abip-sets.h5")
#
#     ags_test, cdrs_test, lbls_test, mask_test = test_set
#     probs_test = model.predict([ags_test, cdrs_test, np.squeeze(mask_test)])
#
#     contact = lbls_test
#     cdrs = mask_test
#     parapred = probs_test
#
#     for name, probs in [("contact", contact), ("CDR", cdrs), ("parapred", parapred)]:
#         annotate_and_save_test_structures(probs, "annotated/" + name)
#
#
# def patchdock_classify():
#     print("CDR results")
#     print(capri_evaluate_test_structures("results/CDR"))
#     # Top 10: {'high': 1, 'med': 2, 'low': 0}
#     # Top 200: {'high': 1, 'med': 14, 'low': 1}
#
#     print("Parapred results")
#     print(capri_evaluate_test_structures("results/parapred"))
#     # Top 10: {'high': 1, 'med': 8, 'low': 0}
#     # Top 200: {'high': 1, 'med': 20, 'low': 2}
#
#     print("Contact results")
#     print(capri_evaluate_test_structures("results/contact"))
#     # Top 10: {'high': 1, 'med': 7, 'low': 1}
#     # Top 200: {'high': 1, 'med': 22, 'low': 3}

if __name__ == "__main__":
    process_cv_results()