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
              validation_split=0.1,
              nb_epoch=20, sample_weight=np.squeeze(labels * 100 + 1))


if __name__ == "__main__":
    main()