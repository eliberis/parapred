from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, \
    Dense, LSTM, Masking, Input, RepeatVector, Dropout, Convolution1D
from keras.layers.merge import concatenate, add
import keras.backend as K
from data_provider import NUM_AA_FEATURES, NEIGHBOURHOOD_SIZE, NUM_EDGE_FEATURES


def false_neg(y_true, y_pred):
    return K.squeeze(K.clip(y_true - K.round(y_pred), 0.0, 1.0), axis=-1)


def false_pos(y_true, y_pred):
    return K.squeeze(K.clip(K.round(y_pred) - y_true, 0.0, 1.0), axis=-1)


# Should probably triple-check that it works as expected
class MaskingByLambda(Layer):
    def __init__(self, func, **kwargs):
        self.supports_masking = True
        self.mask_func = func
        super(MaskingByLambda, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return self.mask_func(input, input_mask)

    def call(self, x, mask=None):
        exd_mask = K.expand_dims(self.mask_func(x, mask), axis=-1)
        return x * K.cast(exd_mask, K.floatx())


# Base masking decision only on the first elements
def mask(input, mask):
    return K.any(K.not_equal(input[:, :, :128], 0.0), axis=-1)


def mask_by_input(tensor):
    return lambda input, mask: tensor


# 1D convolution that supports masking by retaining the mask of the input
class MaskedConvolution1D(Convolution1D):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        assert kwargs['padding'] == 'same' # Only makes sense for 'same'
        super(MaskedConvolution1D, self).__init__(*args, **kwargs)

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def call(self, x, mask=None):
        assert mask is not None
        mask = K.expand_dims(mask, axis=-1)
        x = super(MaskedConvolution1D, self).call(x)
        return x * K.cast(mask, K.floatx())


def get_model(max_ag_len, max_cdr_len):
    nsize_sq = NEIGHBOURHOOD_SIZE ** 2

    ag = Input(shape=(max_ag_len * NEIGHBOURHOOD_SIZE, NUM_AA_FEATURES))
    ag_edges = Input(shape=(max_ag_len * nsize_sq, NUM_EDGE_FEATURES))

    ag_fts = Convolution1D(128, NEIGHBOURHOOD_SIZE,
                           strides=NEIGHBOURHOOD_SIZE,
                           activation='elu')(ag)
    ag_edge_fts = Convolution1D(4, nsize_sq, strides=nsize_sq)(ag_edges)

    ag_all_fts = concatenate([ag_fts, ag_edge_fts])
    ag_fts_m = Masking()(ag_all_fts)

    enc_ag = Bidirectional(LSTM(128,
                                dropout=0.2,
                                recurrent_dropout=0.1),
                           merge_mode='concat')(ag_fts_m)

    cdr = Input(shape=(max_cdr_len * NEIGHBOURHOOD_SIZE, NUM_AA_FEATURES))
    cdr_edges = Input(shape=(max_cdr_len * nsize_sq, NUM_EDGE_FEATURES))

    cdr_fts = Convolution1D(128, NEIGHBOURHOOD_SIZE,
                            strides=NEIGHBOURHOOD_SIZE,
                            activation='elu')(cdr)
    cdr_edge_fts = Convolution1D(4, nsize_sq, strides=nsize_sq)(cdr_edges)

    cdr_all_fts = concatenate([cdr_fts, cdr_edge_fts])
    cdr_fts_m = Masking()(cdr_all_fts)

    ab_net_out = Bidirectional(LSTM(128, return_sequences=True),
                               merge_mode='concat')(cdr_fts_m)

    enc_ag_rep = RepeatVector(max_cdr_len)(enc_ag)
    ab_ag_repr = concatenate([ab_net_out, enc_ag_rep])
    ab_ag_repr = Dropout(0.1)(ab_ag_repr)

    label_mask = Input(shape=(max_cdr_len,))
    ab_ag_repr_m = MaskingByLambda(mask_by_input(label_mask))(ab_ag_repr)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr_m)
    model = Model(inputs=[ag, ag_edges, cdr, cdr_edges, label_mask], outputs=aa_probs)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model
