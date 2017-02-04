from data_provider import load_data_matrices
from model import get_model
import numpy as np

def main():
    examples, labels, ags, params = load_data_matrices()
    print("Examples", examples.shape)
    print("Labels", labels.shape)
    print("Ags", ags.shape)

    model = get_model(params["max_ag_len"], params["max_cdr_len"])
    print(model.summary())

    model.fit([ags, examples], labels, batch_size=32,
              validation_split=0.1, nb_epoch=200,
              sample_weight=np.squeeze(labels * 5 + 1))

    model.save_weights("5.h5")

    # 100: 42s - loss: 1.3170 - binary_accuracy: 0.6649 - precision: 0.2448 - recall: 0.9970 - false_neg: 3.2298e-04 - false_pos: 0.3348 -
    # val_loss: 1.4890 - val_binary_accuracy: 0.6825 - val_precision: 0.2643 - val_recall: 0.9903 - val_false_neg: 0.0011 - val_false_pos: 0.3164

    # 10: 38s - loss: 0.5341 - binary_accuracy: 0.8419 - precision: 0.4019 - recall: 0.9524 - false_neg: 0.0051 - false_pos: 0.1530 -
    # val_loss: 0.6223 - val_binary_accuracy: 0.8380 - val_precision: 0.4073 - val_recall: 0.9103 - val_false_neg: 0.0101 - val_false_pos: 0.1519

    # 8: 40s - loss: 0.4974 - binary_accuracy: 0.8528 - precision: 0.4189 - recall: 0.9435 - false_neg: 0.0061 - false_pos: 0.1411 -
    # val_loss: 0.5897 - val_binary_accuracy: 0.8405 - val_precision: 0.4124 - val_recall: 0.8935 - val_false_neg: 0.0123 - val_false_pos: 0.1472

    # 5: 43s - loss: 0.4162 - binary_accuracy: 0.8641 - precision: 0.4388 - recall: 0.9232 - false_neg: 0.0084 - false_pos: 0.1276 -
    # val_loss: 0.5244 - val_binary_accuracy: 0.8673 - val_precision: 0.4566 - val_recall: 0.7890 - val_false_neg: 0.0243 - val_false_pos: 0.1084


if __name__ == "__main__":
    main()