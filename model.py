import tensorflow as tf
import tensorflow_fold as td
import numpy as np

NUM_FEATS = 21 + 7 # One-hot + 7 features
RNN_STATE_SIZE = 64
CONV_FILTERS = 32
CONV_FILTER_SPAN = 9


def bidirectional_rnn(cell_fwd, cell_bwd):
    rnn = td.Composition()
    with rnn.scope():
        in_seq = td.Identity().reads(rnn.input)
        fwd_pass = td.RNN(cell_fwd).reads(in_seq)
        fwd_seq = fwd_pass[0]
        fwd_final = td.Fold(td.GetItem(1), tf.zeros([RNN_STATE_SIZE])).reads(fwd_pass[0])

        in_seq_rev = td.Slice(step=-1).reads(rnn.input)
        bwd_pass = td.RNN(cell_bwd).reads(in_seq_rev)
        bwd_seq = bwd_pass[0]
        bwd_final = td.Fold(td.GetItem(1), tf.zeros([RNN_STATE_SIZE])).reads(bwd_pass[0])

        concat_seq = td.ZipWith(td.Concat()).reads(fwd_seq, bwd_seq)
        concat_fin_st = td.Concat().reads(fwd_final, bwd_final)
        rnn.output.reads(concat_seq, concat_fin_st)
    return rnn


def pad_input(seq):
    pad_l = [np.zeros((NUM_FEATS, ), dtype='float32')
             for _ in range(CONV_FILTER_SPAN-1)]
    pad_r = [np.zeros((NUM_FEATS, ), dtype='float32')
             for _ in range(CONV_FILTER_SPAN-1)]
    return pad_l + seq + pad_r


def get_model():
    model = td.Composition()
    with model.scope():
        inp_block = \
            td.Record({'ag': td.InputTransform(pad_input) >> td.Map(td.Vector(NUM_FEATS)),
                       'ab': td.InputTransform(pad_input) >> td.Map(td.Vector(NUM_FEATS)),
                       'lb': td.Map(td.Scalar())}).reads(model.input)
        ag_seq = td.GetItem(0).reads(inp_block)
        ab_seq = td.GetItem(1).reads(inp_block)
        labels = td.GetItem(2).reads(inp_block)

        # TODO: add dropout
        # TODO: fix initialisations
        ag_fwd_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.LSTMCell(num_units=RNN_STATE_SIZE))

        ag_bwd_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.LSTMCell(num_units=RNN_STATE_SIZE))

        ag_conv = td.FC(CONV_FILTERS, activation=None)
        ag_sum = (td.NGrams(CONV_FILTER_SPAN) >> # Not 'same', just 'valid'
                  td.Map(td.Concat() >> ag_conv) >>
                  bidirectional_rnn(ag_fwd_lstm_cell, ag_bwd_lstm_cell) >>
                  td.GetItem(1)).reads(ag_seq)
        ab_fwd_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.LSTMCell(num_units=RNN_STATE_SIZE))

        ab_bwd_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.LSTMCell(num_units=RNN_STATE_SIZE))

        ab_p_seq = (bidirectional_rnn(ab_fwd_lstm_cell, ab_bwd_lstm_cell) >>
                    td.GetItem(0)).reads(ab_seq)

        ag_repl = td.Broadcast().reads(ag_sum)
        ab_ag_repr = td.ZipWith(td.Concat()).reads(ab_p_seq, ag_repl)

        dense = td.FC(1, activation=None)
        output_logits = td.Map(td.Function(lambda x: tf.squeeze(x, axis=1)))\
            .reads(td.Map(dense).reads(ab_ag_repr))
        output_probs = td.Map(td.Function(tf.nn.sigmoid)).reads(output_logits)

        # Compute loss (TODO: implement binary x-entropy)
        lossfn = lambda lg, lb: -(lg * lb) + tf.log(1 + tf.exp(lg))
        loss = (td.ZipWith(td.Function(lossfn)) >>
                td.Mean()).reads(output_logits, labels)
        td.Metric('loss').reads(loss)

        output = td.Map(td.Metric('probs')).reads(output_probs)
        no_output = td.Void().reads(output)
        model.output.reads(no_output)
    return model

    # input_ag = Input(shape=(max_ag_len, NUM_FEATS))
    # input_ag_m = Masking()(input_ag)
    #
    # input_ag_conv = MaskedConvolution1D(CONV_FILTERS, CONV_FILTER_SPAN,
    #                                     border_mode='same')(input_ag_m)
    # input_ag_m2 = Masking()(input_ag_conv) # Probably unnecessary, investigate
    #
    # enc_ag = Bidirectional(LSTM(RNN_STATE_SIZE, dropout_U=0.1),
    #                        merge_mode='concat')(input_ag_m2)
    #
    # input_ab = Input(shape=(max_cdr_len, NUM_FEATS))
    # input_ab_m = Masking()(input_ab)
    #
    # # Adding dropout_U here is a bad idea --- sequences are very short and
    # # all information is essential
    # ab_net_out = Bidirectional(LSTM(RNN_STATE_SIZE, return_sequences=True),
    #                            merge_mode='concat')(input_ab_m)
    #
    # enc_ag_rep = RepeatVector(max_cdr_len)(enc_ag)
    # ab_ag_repr = merge([ab_net_out, enc_ag_rep], mode='concat')
    # ab_ag_repr = MaskingByLambda(mask)(ab_ag_repr)
    # ab_ag_repr = Dropout(0.1)(ab_ag_repr)
    #
    # aa_probs = TimeDistributed(Dense(1, activation='sigmoid'))(ab_ag_repr)
    # model = Model(input=[input_ag, input_ab], output=aa_probs)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam', metrics=['binary_accuracy',
    #                                          'precision', 'recall', false_rates],
    #               sample_weight_mode="temporal")
    # return model
