from data_provider import open_dataset, train_test_split, squash_entries_per_loop
from model import get_model
from plotting import *
import numpy as np

def main():
    entries, params = open_dataset()

    num_entries = len(entries)
    max_ag_len = params["max_ag_len"]
    max_cdr_len = params["max_cdr_len"]

    print("Number of samples:", num_entries)
    print("Max AG length:", max_ag_len)
    print("Max CDR length:", max_cdr_len)

    models = {}
    train_set, test_set = train_test_split(entries, test_size=30, seed=0)

    sq_train_set = squash_entries_per_loop(train_set)
    sq_test_set = squash_entries_per_loop(test_set)

    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        print("Training a model for {} CDR:".format(cdr_name))

        model = get_model(params["max_ag_len"], params["max_cdr_len"])
        ags_train, cdrs_train, lbls_train = sq_train_set[cdr_name]

        example_weight = np.squeeze(lbls_train * 5 + 1)  # 6-to-1 in favour of 1
        # model.fit([ags_train, cdrs_train], lbls_train,
        #            batch_size=32, epochs=30,
        #            sample_weight=example_weight)
        model.load_weights(cdr_name + "_weights.h5")
        models[cdr_name] = model

    overall_lbls = []
    overall_preds = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        ags_test, cdrs_test, lbls_test = sq_test_set[cdr_name]
        test_preds = models[cdr_name].predict([ags_test, cdrs_test])
        overall_lbls.append(lbls_test)
        overall_preds.append(test_preds)

    plot_prec_rec_curve(np.concatenate(overall_lbls),
                        np.concatenate(overall_preds),
                        output_filename="overall.png")

    # Only use when examples aren't permuted
    # for i, cdr_name in enumerate(["H1", "H2", "H3", "L1", "L2", "L3"]):
    #     curr_ags = ags_test[i::6]
    #     curr_cdrs = cdrs_test[i::6]
    #     curr_lbls = lbls_test[i::6]
    #     plot_prec_rec_curve(model, curr_ags, curr_cdrs, curr_lbls,
    #                         output_filename=(cdr_name + ".png"))

if __name__ == "__main__":
    main()