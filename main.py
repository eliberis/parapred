from data_provider import *
from structure_processor import *
from model import get_model
from plotting import *
import numpy as np

def main():
    train_set, test_set, params = open_dataset()

    max_ag_len = params["max_ag_len"]
    max_cdr_len = params["max_cdr_len"]
    pos_class_weight = params["pos_class_weight"]

    print("Max AG length:", max_ag_len)
    print("Max CDR length:", max_cdr_len)
    print("Pos class weight:", pos_class_weight)

    model = get_model(params["max_ag_len"], params["max_cdr_len"])
    print(model.summary())

    ags_train, cdrs_train, lbls_train = train_set
    ags_test, cdrs_test, lbls_test = test_set
    example_weight = np.squeeze(lbls_train * 5 + 1)  # 6-to-1 in favour of 1

    history = model.fit([ags_train, cdrs_train], lbls_train,
                        batch_size=32, epochs=30,
                        sample_weight=example_weight)

    model.save_weights("abip-sets.h5")

    plot_prec_rec_curve(model, ags_test, cdrs_test, lbls_test,
                        output_filename="abip-sets.png")

if __name__ == "__main__":
    main()