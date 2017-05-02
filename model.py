from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, \
    Dense, LSTM, Masking, Input, RepeatVector, Dropout, Convolution1D, \
    BatchNormalization, MaxPool1D, Flatten, Activation, Reshape, Lambda
from keras.layers.merge import concatenate, add
import keras.backend as K
from data_provider import NUM_ATOM_FEATURES


AG_RNN_STATE_SIZE = 128
AB_RNN_STATE_SIZE = 128
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
def mask_by_first(num_first):
    return lambda input, mask: \
        K.any(K.not_equal(input[:, :, :num_first], 0.0), axis=-1)


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


class TNet(Layer):
    def __init__(self, localisation_net, point_dim, orthog_loss=0.0,
                 **kwargs):
        self.loc_net = localisation_net
        self.orthog_loss = orthog_loss
        self.point_dim = point_dim
        super().__init__(**kwargs)

    def orthogonal_regularisation_loss(self, mat):
        # Assuming mat is square
        mat_trans = K.permute_dimensions(mat, (0, 2, 1))
        return self.orthog_loss * K.sum(
            K.square(K.eye(self.point_dim) - K.batch_dot(mat, mat_trans)))

    def call(self, x):
        # Could verify shapes here
        transform_mat = self.loc_net(x)
        if self.orthog_loss > 0.0:
            self.add_loss(self.orthogonal_regularisation_loss(transform_mat),
                          inputs=[x])

        result = K.batch_dot(x, transform_mat)
        return result


def loc_net(max_points, point_dim):
    input_points = Input(shape=(max_points, point_dim))

    # Transform each point to get 1024 features
    x = Convolution1D(64, 1, padding='valid', activation='elu')(input_points)
    x = Convolution1D(128, 1, padding='valid', activation='elu')(x)
    x = Convolution1D(1024, 1, padding='valid', activation='elu')(x)

    # Select max of all points
    x = MaxPool1D(pool_size=max_points)(x)
    x = Flatten()(x) # Remove the point dimension

    # Use FCs to reduce to the transformation matrix
    x = Dense(512, activation='elu')(x)
    x = Dense(256, activation='elu')(x)
    x = Dense(point_dim * point_dim)(x)

    # Add an identity matrix
    flat_eye = K.reshape(K.eye(point_dim), (-1, ))
    x = Lambda(lambda x: x + flat_eye)(x)
    transform_mat = Reshape((point_dim, point_dim))(x)
    return Model(inputs=input_points, outputs=transform_mat)


# def point_net(max_points, init_point_dim):
#     input_pts = Input(shape=(max_points, init_point_dim))
#     trans_pts = TNet(loc_net(max_points, init_point_dim), init_point_dim)(input_pts)
#
#     # Note: Not entirely sure what PointNet authors meant here
#     x = Convolution1D(64, 1, padding='valid', activation='elu')(trans_pts)
#     x = Convolution1D(64, 1, padding='valid', activation='elu')(x)
#
#     local_fts = TNet(loc_net(max_points, 64), 64, orthog_loss=0.001)(x)
#
#     x = Convolution1D(64, 1, padding='valid', activation='elu')(local_fts)
#     x = Convolution1D(128, 1, padding='valid', activation='elu')(x)
#     x = Convolution1D(1024, 1, padding='valid', activation='elu')(x)
#
#     x = MaxPool1D(pool_size=max_points)(x)
#     global_feat = Flatten()(x) # Remove the point dimension
#
#     return Model(inputs=input_pts, outputs=global_feat)


def get_model(max_ag_atoms, max_cdr_atoms, max_atoms_per_residue, max_cdr_len):
    # input_ag_atoms = Input(shape=(max_ag_atoms, NUM_ATOM_FEATURES))


    # enc_ag = Bidirectional(LSTM(AG_RNN_STATE_SIZE,
    #                             dropout=0.2,
    #                             recurrent_dropout=0.1),
    #                        merge_mode='concat')(input_ag_m)
    #
    # input_ag_atoms = Input(shape=(max_ag_atoms, NUM_ATOM_FEATURES))
    # global_ag_feat = point_net(max_ag_atoms, NUM_ATOM_FEATURES)(input_ag_atoms)

    # Local CDR features

    ab_pts = Input(shape=(max_cdr_atoms, NUM_ATOM_FEATURES))
    ab_tr_pts = TNet(loc_net(max_cdr_atoms, NUM_ATOM_FEATURES),
                    NUM_ATOM_FEATURES, orthog_loss=0.001)(ab_pts)

    ab_tr_pts = Convolution1D(64, 1, activation='elu')(ab_tr_pts)
    ab_tr_pts = Convolution1D(64, 1, activation='elu')(ab_tr_pts)

    ab_local_fts = TNet(loc_net(max_cdr_atoms, 64),
                        64, orthog_loss=0.001)(ab_tr_pts)

    # Compute global CDR feature

    ab_fts = Convolution1D(64, 1, activation='elu')(ab_local_fts)
    ab_fts = Convolution1D(128, 1, activation='elu')(ab_fts)
    ab_fts = Convolution1D(1024, 1, activation='elu')(ab_fts)

    ab_global_feat = MaxPool1D(pool_size=max_cdr_atoms)(ab_fts)
    ab_global_feat = Flatten()(ab_global_feat)  # Remove the point dimension
    ab_global_feat = RepeatVector(max_cdr_atoms)(ab_global_feat)

    # Local AG features

    ag_pts = Input(shape=(max_ag_atoms, NUM_ATOM_FEATURES))
    ag_tr_pts = TNet(loc_net(max_ag_atoms, NUM_ATOM_FEATURES),
                     NUM_ATOM_FEATURES, orthog_loss=0.001)(ag_pts)

    ag_tr_pts = Convolution1D(64, 1, activation='elu')(ag_tr_pts)
    ag_tr_pts = Convolution1D(64, 1, activation='elu')(ag_tr_pts)

    ag_local_fts = TNet(loc_net(max_ag_atoms, 64),
                        64, orthog_loss=0.001)(ag_tr_pts)

    # Global AG feature
    ag_fts = Convolution1D(64, 1, activation='elu')(ag_local_fts)
    ag_fts = Convolution1D(128, 1, activation='elu')(ag_fts)
    ag_fts = Convolution1D(1024, 1, activation='elu')(ag_fts)

    ag_global_feat = MaxPool1D(pool_size=max_ag_atoms)(ag_fts)
    ag_global_feat = Flatten()(ag_global_feat)  # Remove the point dimension
    ag_global_feat = RepeatVector(max_cdr_atoms)(ag_global_feat)

    # CDR residue probabilities
    all_fts = concatenate([ab_local_fts, ab_global_feat, ag_global_feat])

    fts = Convolution1D(512, 1, activation='elu')(all_fts)
    fts = Convolution1D(256, 1, activation='elu')(fts)
    fts = Convolution1D(128, 1, activation='elu')(fts)

    res_fts = MaxPool1D(pool_size=max_atoms_per_residue,
                        strides=max_atoms_per_residue)(fts)

    neigh_fts = Convolution1D(128, 3, padding='same')(res_fts)

    probs = Convolution1D(1, 1, activation='sigmoid')(neigh_fts)

    # Convolve neighbourhoods here?
    # TODO: add RNN

    # ab_net_out = Bidirectional(LSTM(AB_RNN_STATE_SIZE,
    #                                 return_sequences=True),
    #                            merge_mode='concat')(res_fts)


    # Adding recurrent dropout here is a bad idea
    # --- sequences are very short
    # ab_net_out = Bidirectional(LSTM(AB_RNN_STATE_SIZE,
    #                                 return_sequences=True),
    #                            merge_mode='concat')(input_ab_m)
    #
    # input_ab_atoms = Input(shape=(max_cdr_atoms, NUM_ATOM_FEATURES))
    # global_ab_feat = point_net(max_cdr_atoms, NUM_ATOM_FEATURES)(input_ab_atoms)
    #
    # feats = concatenate([enc_ag, global_ag_feat, global_ab_feat])
    # feats = RepeatVector(max_cdr_len)(feats)
    #
    # ab_ag_repr = concatenate([ab_net_out, feats])
    # ab_ag_repr = MaskingByLambda(ab_mask)(ab_ag_repr)
    # ab_ag_repr = Dropout(0.2)(ab_ag_repr)
    #
    # aa_f = TimeDistributed(Dense(64, activation='elu'))(ab_ag_repr)
    # aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(aa_f)

    model = Model(inputs=[ag_pts, ab_pts], outputs=probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model


def baseline_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len, NUM_CDR_FEATURES))
    input_ab_m = Masking()(input_ab)

    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(input_ab_m)
    model = Model(inputs=input_ab, outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model


def ab_only_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len, NUM_CDR_FEATURES))
    input_ab_m = Masking()(input_ab)

    # Adding dropout_U here is a bad idea --- sequences are very short and
    # all information is essential
    ab_net_out = Bidirectional(LSTM(AB_RNN_STATE_SIZE, return_sequences=True),
                               merge_mode='concat')(input_ab_m)

    ab_ag_repr = Dropout(0.1)(ab_net_out)
    aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    model = Model(inputs=input_ab, outputs=aa_probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model
