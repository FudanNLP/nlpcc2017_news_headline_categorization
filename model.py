import os
import sys

import numpy as np
import tensorflow as tf
import argparse
import logging

import helper
import core_rnn_cell_impl as rnn_cell
from Config import Config
import TfUtils


args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")

    parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
    parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
    parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

    parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
    parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])

    args = parser.parse_args()

class Model():
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, test=False, args=args):
        """options in this function"""
        self.config = Config()

        self.weight_Path = args.weight_path
        if args.load_config == False:
            self.config.saveConfig(self.weight_Path+'/config')
            print 'default configuration generated, please specify --load-config and run again.'
            sys.exit()
        else:
            if os.path.exists(self.weight_Path+'/config'):
                self.config.loadConfig(self.weight_Path+'/config')
            else:
                self.config.saveConfig(self.weight_Path+'/config') #if not exists config file then use default


        self.load_data(test)

        self.add_placeholders()
        inputs = self.add_embedding()
        self.logits = self.add_model(inputs)

        self.predict_prob = tf.nn.softmax(self.logits, name='predict_probability_soft')


        self.loss = self.add_loss_op(self.logits, tf.to_float(self.ph_label))
        self.train_op = self.add_train_op(self.loss)

    def load_data(self, test):
        self.vocab = helper.Vocab()
        self.tag_vocab = helper.Vocab()
        self.vocab.load_vocab_from_file(self.config.vocab_path, sep='\t')
        self.tag_vocab.load_vocab_from_file(self.config.id2tag_path)
        self.config.class_num = len(self.tag_vocab)
        if test==False:
            self.val_data_y, self.val_data_len, self.val_data_x = helper.mkDataSet(self.config.val_data,
                                                self.config.class_num, self.vocab, self.tag_vocab, self.config.num_steps)
            self.test_data_y, self.test_data_len, self.test_data_x = helper.mkDataSet(self.config.test_data,
                                                self.config.class_num, self.vocab, self.tag_vocab, self.config.num_steps)
            self.train_data_y, self.train_data_len, self.train_data_x = helper.mkDataSet(self.config.train_data,
                                                self.config.class_num, self.vocab, self.tag_vocab, self.config.num_steps)

    def add_placeholders(self):
        """
        Adds placeholder variables to tensorflow computational graph.
            self.ph_input: shape(batch_size, sent_len)
            self.ph_label: shape(batch_size, class_num)
            self.ph_seqLen : shape(batch_size)
            self.ph_drop: scalar, dropout(keep rate)

        """
        self.ph_input = tf.placeholder(tf.int32, (None, self.config.num_steps))
        self.ph_label = tf.placeholder(tf.int32, (None, self.config.class_num))
        self.ph_seqLen = tf.placeholder(tf.int32, (None,))
        self.ph_drop = tf.placeholder(tf.float32)

    def create_feed_dict(self, input_batch, seqLen_batch, label_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:

        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
              tensors created in add_placeholders.

        Args:
          input_batch: A batch of input data.
          seqLen_batch: the length of sentence, shape(batch_size)
          label_batch: A batch of label data.
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """

        if label_batch is None:
            holder_list = [self.ph_input, self.ph_seqLen, self.ph_drop]
            feed_list = (input_batch, seqLen_batch, self.config.dropout)
        else:
            holder_list = [self.ph_input, self.ph_label, self.ph_seqLen, self.ph_drop]
            feed_list = (input_batch, label_batch, seqLen_batch, self.config.dropout)

        feed_dict = dict(zip(holder_list, feed_list))
        return feed_dict

    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.

        Returns:
            inputs: shape(b_sz, tstp, emb_sz), fetched input
        """
        if self.config.pre_trained:
            embed_dic = helper.readEmbedding(self.config.embed_path+str(self.config.embed_size))  #embedding.50 for 50 dim embedding
            embed_matrix = helper.mkEmbedMatrix(embed_dic, self.vocab.word_to_index)
            self.embedding = tf.Variable(embed_matrix, 'Embedding')
        else:
            self.embedding = tf.get_variable(
              'Embedding',
              [len(self.vocab), self.config.embed_size], trainable=True)
        inputs = tf.nn.embedding_lookup(self.embedding, self.ph_input)  # shape(b_sz, tstp, emb_sz)
        return inputs

    def add_model(self, inputs):
        """Implements core of model that transforms input_data into predictions.

        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Models can be added by defining another function like that have defined below
        (take `inputs` and generate `logtis`),
        and add two lines of control statement:
            if self.config.neural_model == 'yourModel':
                logits = basic_lstm_model(inputs)
        then you can specify which model to use in your config file.

        Args:
          inputs: a tensor have the shape of shape(b_sz, tstp, emb_sz)
        Returns:
          logits: A tensor take the shape of shape(batch_size, n_classes), score tensor
        """
        input_shape = tf.shape(inputs)
        b_sz = input_shape[0]
        tstp = input_shape[1]
        emb_sz = self.config.embed_size

        def basic_lstm_model(inputs):
            print "Loading basic lstm model.."
            for i in range(self.config.rnn_numLayers):
                with tf.variable_scope('rnnLayer'+str(i)):
                    lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
                    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, self.ph_seqLen,  #(b_sz, tstp, h_sz)
                                                   dtype=tf.float32 ,swap_memory=True,
                                                   scope = 'basic_lstm_model_layer-'+str(i))
                    inputs = outputs #b_sz, tstp, h_sz
            mask = TfUtils.mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            mask = tf.expand_dims(mask, axis=2) #b_sz, tstp, 1

            aggregate_state = TfUtils.reduce_avg(outputs, self.ph_seqLen, dim=1) #b_sz, h_sz
            inputs = aggregate_state
            inputs = tf.reshape(inputs, [-1, self.config.hidden_size])

            for i in range(self.config.fnn_numLayers):
                inputs = TfUtils.linear(inputs, self.config.hidden_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = TfUtils.linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits

        def basic_cbow_model(inputs):
            mask = TfUtils.mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            mask = tf.expand_dims(mask, axis=2) #b_sz, tstp, 1

            aggregate_state = TfUtils.reduce_avg(inputs, self.ph_seqLen, dim=1) #b_sz, emb_sz
            inputs = aggregate_state
            inputs = tf.reshape(inputs, [-1, self.config.embed_size])

            for i in range(self.config.fnn_numLayers):
                inputs = TfUtils.linear(inputs, self.config.embed_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = TfUtils.linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits

        def basic_cnn_model(inputs):
            in_channel = self.config.embed_size
            filter_sizes = self.config.filter_sizes
            out_channel = self.config.num_filters
            input = inputs
            for layer in range(self.config.cnn_numLayers):
                with tf.name_scope("conv-layer-"+ str(layer)):
                    conv_outputs = []
                    for i, filter_size in enumerate(filter_sizes):
                        with tf.variable_scope("conv-maxpool-%d" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, in_channel, out_channel]
                            W = tf.get_variable(name='W', shape=filter_shape)
                            b = tf.get_variable(name='b', shape=[out_channel])
                            conv = tf.nn.conv1d(                # size (b_sz, tstp, out_channel)
                              input,
                              W,
                              stride=1,
                              padding="SAME",
                              name="conv")
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            conv_outputs.append(h)
                    input = tf.concat(axis=2, values=conv_outputs) #b_sz, tstp, out_channel*len(filter_sizes)
                    in_channel = out_channel * len(filter_sizes)
            # Maxpooling
#             mask = tf.sequence_mask(self.ph_seqLen, tstp, dtype=tf.float32) #(b_sz, tstp)
            mask = TfUtils.mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            pooled = tf.reduce_max(input*tf.expand_dims(tf.cast(mask, dtype=tf.float32), 2), [1]) #(b_sz, out_channel*len(filter_sizes))
            #size (b_sz, out_channel*len(filter_sizes))
            inputs = tf.reshape(pooled, shape=[b_sz, out_channel*len(filter_sizes)])

            for i in range(self.config.fnn_numLayers):
                inputs = TfUtils.linear(inputs, self.config.embed_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = TfUtils.linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits

        if self.config.neural_model == 'lstm_basic':
            logits = basic_lstm_model(inputs)
        elif self.config.neural_model == 'cbow_basic':
            logits = basic_cbow_model(inputs)
        elif self.config.neural_model == 'cnn_basic':
            logits = basic_cnn_model(inputs)
        else:
            raise ValueError('No such model:'+ self.config.neural_model)

        return logits

    def add_loss_op(self, logits, labels):
        """Adds ops for loss to the computational graph.

        Args:
          logits: A tensor of shape (batch_size, n_classes)
          labels: A tensor - placeholder probably,  of shape (batch_size, n_class)
        Returns:
          loss: A 0-d tensor (scalar) output
        """
        labels_float = tf.to_float(labels)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_float)
        loss = tf.reduce_mean(loss)
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding])
        loss = loss + self.config.reg * reg_loss
        return loss

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   self.config.decay_steps, self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def run_epoch(self, sess, data_x, data_y, len_list, verbose=10):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            data_x: input data, have shape of (data_num, num_steps), change it to ndarray before this function is called
            data_y: label, have shape of (data_num, class_num)
            len_list: length list correspond to data_x, have shape of (data_num)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(data_x)
        total_steps =data_len // self.config.batch_size
        total_loss = []
        for step, indices in enumerate(helper.data_iter_indices(data_len, self.config.batch_size)):
            feed_dict = self.create_feed_dict(data_x[indices], len_list[indices], data_y[indices])
            _, loss, lr = sess.run([self.train_op, self.loss, self.learning_rate], feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        return np.mean(total_loss)

    def fit(self, sess, data_x, data_y, len_list, verbose=10):

        data_len = len(data_x)
        total_loss = []
        for step, indices in enumerate(helper.data_iter_indices(data_len, self.config.batch_size)):
            feed_dict = self.create_feed_dict(data_x[indices], len_list[indices], data_y[indices])
            loss = sess.run(self.loss, feed_dict=feed_dict)
            total_loss.append(loss)
        return np.mean(total_loss)

    def predict(self, sess, data_x, len_list):
        """Make predictions from the provided model.
        Args:
            sess: tf.Session() obj
            data_x: input data matrix have the shape of (data_num, num_steps), change it to ndarray before this function is called
            len_list: input data_length have the shape of (data_num)
        Returns:
          ret_pred_prob: Probability of the prediction with respect to each class
        """
        ret_pred_prob = []
        for indices in helper.pred_data_iter_indices(len(data_x), self.config.batch_size):
            feed_dict = self.create_feed_dict(data_x[indices], len_list[indices])
            pred_prob = sess.run(self.predict_prob, feed_dict=feed_dict)
            ret_pred_prob.append(pred_prob)
        ret_pred_prob = np.concatenate(ret_pred_prob, axis=0)
        return ret_pred_prob

    ###################################################################################################

    """complementay predict"""

    def predict_label(self, sess, data_in, label_num):
        def get_class_serious_id_map(vocab):
            def makeMap(dict_input):
                label = [o[0] for o in dict_input.items()]
                map_id = [o[1] for o in dict_input.items()]
                return label, map_id
            class_ids = {}
            serious_ids={}
            for item in vocab.word_to_index:
                if item == vocab.unknown:
                    class_ids[item] = [vocab.encode(item)]
                    continue
                class_name = ','.join(item.split(',')[:-1])
                serious = item.split(',')[-1]
                if class_name not in class_ids:
                    class_ids[class_name] = [vocab.encode(item)]
                else:
                    class_ids[class_name].append(vocab.encode(item))

                if serious not in serious_ids:
                    serious_ids[serious] = [vocab.encode(item)]
                else:
                    serious_ids[serious].append(vocab.encode(item))
            return makeMap(class_ids), makeMap(serious_ids)

        def fetch_label_prob(label_map_ids, prob_matrix):
            '''assume that prob_matrix is a ndarray'''
            label_list = label_map_ids[0]
            ids_list = label_map_ids[1]

            label_prob = [np.sum(prob_matrix[:, o], axis=1) for o in ids_list] #(label_num, batch_sz)
            label_prob = np.array(label_prob)
            right_id = np.argmax(label_prob, axis=0) #(b_sz)
            ret_label = [label_list[idx] for idx in right_id]
            ret_prob = label_prob[right_id, range(len(prob_matrix))] #b_sz
            return zip(ret_label, ret_prob.tolist())

        len_list, data_x = helper.encodeNpad(data_in, self.vocab, self.config.num_steps)
        data_x = np.array(data_x)
        len_list = np.array(len_list)
        prob_matrix = self.predict(sess, data_x, len_list)

        order = np.argsort(prob_matrix,axis=1)
        ret_tuple = []
        for i in range(len(data_in)):
            dummy = [self.tag_vocab.decode(id) for id in order[i][-label_num[i]:]]
            dummy_prob = [prob_matrix[i][id] for id in order[i][-label_num[i]:]]
            ret_item = zip(dummy, dummy_prob) #(label_num, 2)
            ret_tuple.append(ret_item) # b_sz, label_num, 2
        return ret_tuple

def test_case(sess, classifier, data_x, data_y, data_len, onset='VALIDATION'):
    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    loss = classifier.fit(sess, data_x, data_y, data_len)
    pred_prob = classifier.predict(sess, data_x, data_len)
    pred = helper.pred_from_prob_multi(pred_prob, np.sum(data_y, axis=1))  # (data_num, class_num)
    prec, recall, overall_prec, overall_recall = helper.calculate_confusion_multi(pred, data_y)
    helper.print_confusion_multi(prec, recall, overall_prec, overall_recall, classifier.tag_vocab.index_to_word)
    accuracy = helper.calculate_accuracy_multi(pred, data_y)

    print 'Overall '+onset+' accuracy is: {}'.format(accuracy)
    logging.info('Overall '+onset+' accuracy is: {}'.format(accuracy))
    print 'Overall ' + onset + ' loss is: {}'.format(loss)
    logging.info('Overall ' + onset + ' loss is: {}'.format(loss))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    return accuracy, loss

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            classifier = Model()
        saver = tf.train.Saver()

        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_accuracy = 0
            best_val_epoch = 0
            sess.run(tf.global_variables_initializer())

            train_data_x = np.array(classifier.train_data_x)
            train_data_y = np.array(classifier.train_data_y)
            train_data_len = np.array(classifier.train_data_len)

            val_data_x = np.array(classifier.val_data_x)
            val_data_y = np.array(classifier.val_data_y)
            val_data_len = np.array(classifier.val_data_len)

            for epoch in range(classifier.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = classifier.run_epoch(sess, train_data_x, train_data_y, train_data_len)
                print
                print "Mean loss in this epoch is: ", loss
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss) )
                print '='*50

                if args.debug_enable:
                    test_case(sess, classifier, train_data_x, train_data_y, train_data_len, onset='TRAINING')
                val_accuracy, loss = test_case(sess, classifier, val_data_x, val_data_y, val_data_len, onset='VALIDATION')

                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(classifier.weight_Path):
                        os.makedirs(classifier.weight_Path)

                    saver.save(sess, classifier.weight_Path+'/classifier.weights')
                if epoch - best_val_epoch > classifier.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
    logging.info("Training complete")

def test_run():

    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):   #gpu_num options
            classifier = Model()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, classifier.weight_Path+'/classifier.weights')

            test_data_x = np.array(classifier.test_data_x)
            test_data_y = np.array(classifier.test_data_y)
            test_data_len = np.array(classifier.test_data_len)

            accu, loss = test_case(sess, classifier, test_data_x, test_data_y, test_data_len, onset='TEST')

def main(_):
    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    logFile = args.weight_path+'/run.log'

    if args.train_test == "train":

        try:
            os.remove(logFile)
        except OSError:
            pass
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        train_run()
    else:
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        test_run()

if __name__ == '__main__':
    tf.app.run()
