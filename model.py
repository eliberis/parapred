from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, \
    Dense, LSTM, Masking, Input, RepeatVector, Dropout, Convolution1D
from keras.layers.merge import concatenate
import keras.backend as K
from data_provider import NUM_FEATURES

RNN_STATE_SIZE = 64
CONV_FILTERS = 32
CONV_FILTER_SPAN = 9


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
    return K.any(K.not_equal(input[:, :, :(2*RNN_STATE_SIZE)], 0.0), axis=-1)


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
    input_ag = Input(shape=(max_ag_len, NUM_FEATURES))
    input_ag_m = Masking()(input_ag)

    input_ag_conv = MaskedConvolution1D(CONV_FILTERS, CONV_FILTER_SPAN,
                                        padding='same')(input_ag_m)
    input_ag_m2 = Masking()(input_ag_conv) # Probably unnecessary, investigate

    enc_ag = Bidirectional(LSTM(RNN_STATE_SIZE, dropout=0.1,
                                recurrent_dropout=0.1),
                           merge_mode='concat')(input_ag_m2)

    input_ab = Input(shape=(max_cdr_len, NUM_FEATURES))

    ab_fts = Convolution1D(32, 1, activation='elu')(input_ab)
    ab_seq = Masking()(ab_fts)
    ab_neigh_fts = MaskedConvolution1D(32, 3, padding='same', activation='elu')(ab_seq)

    label_mask = Input(shape=(max_cdr_len,))

    # Adding dropout_U here is a bad idea --- sequences are very short and
    # all information is essential
    ab_net_out = Bidirectional(LSTM(RNN_STATE_SIZE, return_sequences=True),
                               merge_mode='concat')(ab_neigh_fts)

    enc_ag_rep = RepeatVector(max_cdr_len)(enc_ag)
    ab_ag_repr = concatenate([ab_net_out, enc_ag_rep])
    ab_ag_repr = MaskingByLambda(mask_by_input(label_mask))(ab_ag_repr)
    ab_ag_repr = Dropout(0.1)(ab_ag_repr)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    model = Model(inputs=[input_ag, input_ab, label_mask], outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model


def baseline_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len, NUM_FEATURES))
    input_ab_m = Masking()(input_ab)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(input_ab_m)
    model = Model(inputs=input_ab, outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model


def ab_only_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len, NUM_FEATURES))
    input_ab_m = Masking()(input_ab)

    # Adding dropout_U here is a bad idea --- sequences are very short and
    # all information is essential
    ab_net_out = Bidirectional(LSTM(RNN_STATE_SIZE, return_sequences=True),
                               merge_mode='concat')(input_ab_m)

    ab_ag_repr = Dropout(0.1)(ab_net_out)
    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    model = Model(inputs=input_ab, outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model
