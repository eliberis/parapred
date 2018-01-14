from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, \
    Dense, LSTM, Masking, Input, RepeatVector, Dropout, Convolution1D, \
    BatchNormalization, Activation
from keras.layers.merge import concatenate, add
import keras.backend as K
from keras.regularizers import l2
from data_provider import NUM_FEATURES


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


def ab_ag_seq_model(max_ag_len, max_cdr_len):
    input_ag = Input(shape=(max_ag_len, NUM_FEATURES))
    ag_seq = Masking()(input_ag)

    enc_ag = Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1),
                           merge_mode='concat')(ag_seq)

    input_ab = Input(shape=(max_cdr_len, NUM_FEATURES))
    label_mask = Input(shape=(max_cdr_len,))

    seq = Masking()(input_ab)

    loc_fts = MaskedConvolution1D(64, 5, padding='same', activation='elu')(seq)

    glb_fts = Bidirectional(LSTM(256, dropout=0.15, recurrent_dropout=0.2,
                                 return_sequences=True),
                            merge_mode='concat')(loc_fts)

    enc_ag_rep = RepeatVector(max_cdr_len)(enc_ag)
    ab_ag_repr = concatenate([glb_fts, enc_ag_rep])
    ab_ag_repr = MaskingByLambda(mask_by_input(label_mask))(ab_ag_repr)
    ab_ag_repr = Dropout(0.3)(ab_ag_repr)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    model = Model(inputs=[input_ag, input_ab, label_mask], outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model


def base_ab_seq_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len, NUM_FEATURES))
    label_mask = Input(shape=(max_cdr_len,))

    seq = MaskingByLambda(mask_by_input(label_mask))(input_ab)
    loc_fts = MaskedConvolution1D(28, 3, padding='same', activation='elu',
                                  kernel_regularizer=l2(0.01))(seq)

    res_fts = add([seq, loc_fts])

    glb_fts = Bidirectional(LSTM(256, dropout=0.15, recurrent_dropout=0.2,
                                 return_sequences=True),
                            merge_mode='concat')(res_fts)

    fts = Dropout(0.3)(glb_fts)
    probs = TimeDistributed(Dense(1, activation='sigmoid',
                                  kernel_regularizer=l2(0.01)))(fts)
    return input_ab, label_mask, res_fts, probs


def ab_seq_model(max_cdr_len):
    input_ab, label_mask, _, probs = base_ab_seq_model(max_cdr_len)
    model = Model(inputs=[input_ab, label_mask], outputs=probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model


def conv_output_ab_seq_model(max_cdr_len):
    input_ab, label_mask, loc_fts, probs = base_ab_seq_model(max_cdr_len)
    model = Model(inputs=[input_ab, label_mask], outputs=[probs, loc_fts])
    return model
