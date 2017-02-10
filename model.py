from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, \
    Dense, LSTM, Masking, Input, merge, RepeatVector, Dropout
import keras.backend as K

NUM_FEATS = 21 + 7 # One-hot + 7 features
RNN_STATE_SIZE = 64

def false_rates(y_true, y_pred):
    false_neg = K.mean(K.clip(y_true - K.round(y_pred), 0.0, 1.0))
    false_pos = K.mean(K.clip(K.round(y_pred) - y_true, 0.0, 1.0))
    return {
        'false_neg': false_neg,
        'false_pos': false_pos
    }

# Should probably triple-check that it works as expected
class MaskingByLambda(Layer):
    def __init__(self, func, **kwargs):
        self.supports_masking = True
        self.mask_func = func
        super(MaskingByLambda, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return self.mask_func(input, input_mask)

    def call(self, x, mask=None):
        exd_mask = K.expand_dims(self.mask_func(x, mask), dim=-1)
        return x * K.cast(exd_mask, K.floatx())

# Base masking decision only on the first elements
def mask(input, mask):
    return K.any(K.not_equal(input[:, :, :(2*RNN_STATE_SIZE)], 0.0), axis=-1)

def get_model(max_ag_len, max_cdr_len):
    input_ag = Input(shape=(max_ag_len, NUM_FEATS))
    input_ag_m = Masking()(input_ag)
    enc_ag = Bidirectional(LSTM(RNN_STATE_SIZE, dropout_U=0.1),
                           merge_mode='concat')(input_ag_m)

    input_ab = Input(shape=(max_cdr_len, NUM_FEATS))
    input_ab_m = Masking()(input_ab)

    # Adding dropout_U here is a bad idea --- sequences are very short and
    # all information is essential
    ab_net_out = Bidirectional(LSTM(RNN_STATE_SIZE, return_sequences=True),
                               merge_mode='concat')(input_ab_m)

    enc_ag_rep = RepeatVector(max_cdr_len)(enc_ag)
    ab_ag_repr = merge([ab_net_out, enc_ag_rep], mode='concat')
    ab_ag_repr = MaskingByLambda(mask)(ab_ag_repr)
    ab_ag_repr = Dropout(0.1)(ab_ag_repr)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    model = Model(input=[input_ag, input_ab], output=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy',
                                             'precision', 'recall', false_rates],
                  sample_weight_mode="temporal")
    return model
