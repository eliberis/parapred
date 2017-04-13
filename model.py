from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, Lambda, \
    Dense, LSTM, Masking, Input, RepeatVector, Dropout, Convolution1D
from keras.layers.merge import concatenate
import keras.backend as K
from data_provider import NUM_FEATURES
from data_provider import NUM_CDR_FEATURES, NUM_AG_FEATURES

RNN_STATE_SIZE = 64

RNN_STATE_SIZE = 128
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
    input_ag = Input(shape=(max_ag_len, NUM_AG_FEATURES))
    input_ab = Input(shape=(6, max_cdr_len, NUM_FEATURES))

    # Split input_ab into 6 sub-inputs
    input_loops = [Lambda(lambda x: x[:, i, :, :],
                          output_shape=(max_cdr_len, NUM_FEATURES))(input_ab)
                   for i in range(6)]

    conv1d = MaskedConvolution1D(CONV_FILTERS, CONV_FILTER_SPAN, padding='same')
    masked_loops = [Masking()(inp) for inp in input_loops]

    ab_net = Bidirectional(LSTM(RNN_STATE_SIZE), merge_mode='concat')
    enc_ab = concatenate([ab_net(conv1d(inp)) for inp in masked_loops])

    input_ag = Input(shape=(max_ag_len, NUM_FEATURES))
    input_ag_m = Masking()(input_ag)

    # input_ag_conv = MaskedConvolution1D(CONV_FILTERS, CONV_FILTER_SPAN,
    #                                     padding='same')(input_ag_m)
    # input_ag_m2 = Masking()(input_ag_conv) # Probably unnecessary, investigate
    ag_repr = Bidirectional(LSTM(RNN_STATE_SIZE, return_sequences=True),
                            merge_mode='concat')(input_ag_m)

    enc_ag = Bidirectional(LSTM(RNN_STATE_SIZE,
                                dropout=0.1,
                                recurrent_dropout=0.1),
                           merge_mode='concat')(input_ag_m)
    enc_ab_rep = RepeatVector(max_ag_len)(enc_ab)
    ag_ab_repr = concatenate([ag_repr, enc_ab_rep])
    ag_ab_repr = MaskingByLambda(mask)(ag_ab_repr)
    ag_ab_repr = Dropout(0.1)(ag_ab_repr)

    input_ab = Input(shape=(max_cdr_len, NUM_CDR_FEATURES))
    input_ab_m = Masking()(input_ab)

    # Adding recurrent dropout here is a bad idea
    # --- sequences are very short
    ab_net_out = Bidirectional(LSTM(RNN_STATE_SIZE, return_sequences=True),
                               merge_mode='concat')(input_ab_m)

    enc_ag_rep = RepeatVector(max_cdr_len)(enc_ag)
    ab_ag_repr = concatenate([ab_net_out, enc_ag_rep])
    ab_ag_repr = MaskingByLambda(mask)(ab_ag_repr)
    ab_ag_repr = Dropout(0.1)(ab_ag_repr)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    model = Model(inputs=[input_ag, input_ab], outputs=aa_probs)
    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ag_ab_repr)
    model = Model(inputs=[input_ab, input_ag], outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model
