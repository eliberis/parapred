from data_provider import *
from evaluation import *
from model import *
from plotting import *
from keras.callbacks import LearningRateScheduler, EarlyStopping
import numpy as np
from os import makedirs
from os.path import isfile


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


def full_run(dataset="data/sabdab_27_jun_95_90.csv", out_weights="weights.h5"):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]

    sample_weight = np.squeeze((lbls * 1.5 + 1) * masks)
    model = ab_seq_model(dataset["max_cdr_len"])

    rate_schedule = lambda e: 0.001 if e >= 10 else 0.01

    model.fit([cdrs, np.squeeze(masks)],
              lbls, batch_size=32, epochs=16,
              sample_weight=sample_weight,
              callbacks=[LearningRateScheduler(rate_schedule)])

    model.save_weights(out_weights)


def run_cv(dataset="data/sabdab_27_jun_95_90.csv",
           output_folder="cv-ab-seq",
           num_iters=10):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)
    model_factory = lambda: ab_seq_model(dataset["max_cdr_len"])

    makedirs(output_folder + "/weights", exist_ok=True)
    iters = range(num_iters) if type(num_iters) is int else range(*num_iters)
    for i in iters:
        print("Crossvalidation run", i+1)
        output_file = "{}/run-{}.p".format(output_folder, i)
        weights_template = output_folder + "/weights/run-" + \
                           str(i) + "-fold-{}.h5"
        kfold_cv_eval(model_factory, dataset,
                      output_file, weights_template, seed=i)


def process_cv_results(cv_result_folder="runs/cv-full-2Jan",
                       abip_result_folder="runs/cv-full-ab-2Jan"):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"

    for i, loop in enumerate(["H1", "H2", "H3", "L1", "L2", "L3"]):
        print("Classifier metrics for loop type", loop)
        labels, probs = open_crossval_results(cv_result_folder, 10, i)
        compute_classifier_metrics(labels, probs)

    # Plot PR curves
    print("Plotting PR curves")
    labels, probs = open_crossval_results(cv_result_folder, 10)
    labels_abip, probs_abip = open_crossval_results(abip_result_folder, 10)

    fig = plot_pr_curve(labels, probs, colours=("#0072CF", "#68ACE5"),
                        label="Parapred")
    fig = plot_pr_curve(labels_abip, probs_abip, colours=("#D6083B", "#EB99A9"),
                        label="Parapred using ABiP data", plot_fig=fig)
    fig = plot_abip_pr(fig)
    fig.savefig("pr.eps")

    # Computing overall classifier metrics
    print("Computing classifier metrics")
    compute_classifier_metrics(labels, probs)


def plot_dataset_fraction_results(baseline, d60, d80, dfull):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"

    print("Plotting PR curves")
    labels_baseline, probs_baseline = open_crossval_results(baseline, 10)
    labels_d60, probs_d60 = open_crossval_results(d60, 10)
    labels_d80, probs_d80 = open_crossval_results(d80, 10)
    labels_dfull, probs_dfull = open_crossval_results(dfull, 10)

    fig = plot_pr_curve(labels_dfull, probs_dfull, colours=("#0072CF", "#68ACE5"),
                        label="Parapred (239 entries)")
    fig = plot_pr_curve(labels_d80, probs_d80, colours=("#EA7125", "#F3BD48"),
                        label="Parapred (80% of data, 191 entries)", plot_fig=fig)
    fig = plot_pr_curve(labels_d60, probs_d60, colours=("#55A51C", "#AAB300"),
                        label="Parapred (60% of data, 143 entries)", plot_fig=fig)
    fig = plot_pr_curve(labels_baseline, probs_baseline, colours=("#D6083B", "#EB99A9"),
                        label="Parapred using ABiP data (148 entries)", plot_fig=fig)
    fig.savefig("fractions-pr.eps")


def patchdock_prepare():
    model_weights = "dock-weights.h5"
    if not isfile(model_weights):
        full_run("data/dock_train.csv", model_weights)

    dataset = open_dataset("data/dock_test.csv", "dock_test.csv.p")
    model = ab_seq_model(dataset["max_cdr_len"])
    model.load_weights(model_weights)

    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]
    probs = model.predict([cdrs, np.squeeze(masks)])

    contact = lbls
    cdrs = masks
    parapred = probs

    for name, probs in [("contact", contact), ("CDR", cdrs), ("parapred", parapred)]:
        print("Annotating structures with {} data".format(name))
        makedirs("annotated/" + name)
        annotate_and_save_test_structures("data/dock_test.csv", probs,
                                          "annotated/" + name)


def patchdock_classify():
    # Strip everything but coordinates from PatchDock output using:
    # for f in *; do cat $f | cut -d '|' -f 14- |  grep '^ '  |
    #     grep -v "Ligand" | head -n 200 > $f; done

    print("CDR results")
    print(capri_evaluate_test_structures("data/dock_test.csv", "results/CDR"))

    print("Contact results")
    print(capri_evaluate_test_structures("data/dock_test.csv", "results/contact"))

    print("Parapred results")
    print(capri_evaluate_test_structures("data/dock_test.csv", "results/parapred"))


def show_binding_profiles(run):
    labels, probs = open_crossval_results(run, flatten_by_lengths=False)
    labels = labels[0]  # Labels are constant, any of the 10 runs would do
    probs = np.stack(probs).mean(axis=0)  # Mean binding probability across runs

    contact = binding_profile("data/sabdab_27_jun_95_90.csv", labels)
    print("Contact per-residue binding profile:")
    total = sum(list(contact.values()))
    contact = {k: v / total for k, v in contact.items()}
    print(contact)

    parapred = binding_profile("data/sabdab_27_jun_95_90.csv", probs)
    print("Model's predictions' per-residue binding profile:")
    total = sum(list(parapred.values()))
    parapred = {k: v / total for k, v in parapred.items()}
    print(parapred)

    plot_binding_profiles(contact, parapred)


def evaluate(test_dataset, weights="weights.h5"):
    cache_file = test_dataset.split("/")[-1] + ".p"
    dataset = open_dataset(test_dataset, dataset_cache=cache_file)
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]

    model = ab_seq_model(dataset["max_cdr_len"])
    model.load_weights(weights)
    probs = model.predict([cdrs, np.squeeze(masks)])

    seq_lens = np.sum(np.squeeze(masks), axis=1)
    p = flatten_with_lengths(probs, seq_lens)
    l = flatten_with_lengths(lbls, seq_lens)

    compute_classifier_metrics([l], [p])


def print_neighbourhood_tops(weights="weights.h5"):
    tops = neighbourhood_tops(weights, top_k=10, num_filters_first=20)

    print("Top 10 sequences activating first 20 conv. filters:")
    for f in range(20):
        print("F{}".format(f), end='\t')
    print()

    for i in range(10):
        for f in range(20):
            print(tops[f][i], end='\t')
        print()
    print()


if __name__ == "__main__":
    full_run()
