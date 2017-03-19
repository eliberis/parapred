import tensorflow as tf
import tensorflow_fold as td
import numpy as np

NUM_FEATS = 21 + 7 # One-hot + 7 features
RNN_STATE_SIZE = 64
CONV_FILTERS = 32
CONV_FILTER_SPAN = 9
POS_WEIGHT = 6  # positive samples are weighted 6 times more


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
    pad_len = (CONV_FILTER_SPAN - 1) // 2
    pad_l = [np.zeros((NUM_FEATS, ), dtype='float32')
             for _ in range(pad_len)]
    pad_r = [np.zeros((NUM_FEATS, ), dtype='float32')
             for _ in range(pad_len)]
    return pad_l + seq + pad_r


def conv1d(num_filters, filter_span):
    return (td.NGrams(filter_span) >>
            td.Map(td.Concat() >> td.FC(num_filters, activation=None)))


def get_model(keep_prob=1.0):
    model = td.Composition()
    with model.scope():
        inp_block = \
            td.Record({'ab': td.Map(td.Vector(NUM_FEATS)),
                       'ag': td.InputTransform(pad_input) >>
                             td.Map(td.Vector(NUM_FEATS)),
                       'lb': td.Map(td.Scalar())}).reads(model.input)
        ab_seq = td.GetItem(0).reads(inp_block)  # Alphabetical!
        ag_seq = td.GetItem(1).reads(inp_block)
        labels = td.GetItem(2).reads(inp_block)

        # TODO: add dropout
        # TODO: fix initialisations
        ag_fwd_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(num_units=RNN_STATE_SIZE),
            input_keep_prob=keep_prob, output_keep_prob=keep_prob))

        ag_bwd_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(num_units=RNN_STATE_SIZE),
            input_keep_prob=keep_prob, output_keep_prob=keep_prob))

        ag_sum = (conv1d(CONV_FILTERS, CONV_FILTER_SPAN) >>
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

        fc = td.FC(1, activation=None, input_keep_prob=keep_prob)
        logits = (td.Map(fc) >>
                  td.Map(td.Function(lambda x: tf.squeeze(x, axis=1))))\
            .reads(ab_ag_repr)
        probs = td.Map(td.Function(tf.nn.sigmoid)).reads(logits)

        # Compute loss (TODO: implement binary x-entropy properly)
        lossfn = lambda lg, lb: -(lg * lb) + tf.log(1 + tf.exp(lg))
        losses = td.ZipWith(td.Function(lossfn)).reads(logits, labels)

        scale_loss = lambda loss, lb: loss * (POS_WEIGHT * lb + 1)
        w_losses = td.ZipWith(td.Function(scale_loss)).reads(losses, labels)

        mean_loss = td.Mean().reads(w_losses)
        td.Metric('loss').reads(mean_loss)

        output = td.Map(td.Metric('probs')).reads(probs)
        no_output = td.Void().reads(output)
        model.output.reads(no_output)
    return model
