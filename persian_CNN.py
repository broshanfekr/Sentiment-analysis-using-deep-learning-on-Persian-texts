import tensorflow as tf
import os


class TextCNN(object):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.weigth_norm = l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.correct_predictions = correct_predictions
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.saver = tf.train.Saver(max_to_keep=40)
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)  # .minimize(self.loss)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # self.current_step = tf.train.global_step(self.sess, self.global_step)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def partial_fit(self, x_batch, y_batch, dropout_keep_prob):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.dropout_keep_prob: dropout_keep_prob}

        _, step, loss, accuracy, correct_predict, weigth_norm = self.sess.run(
            [self.train_op, self.global_step, self.loss, self.accuracy, self.correct_predictions, self.weigth_norm],
            feed_dict)
        return step, weigth_norm, loss, accuracy

    def test(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0
        }
        loss, accuracy, correct_predict, predictions = self.sess.run(
            [self.loss, self.accuracy, self.correct_predictions, self.predictions],
            feed_dict)
        return loss, accuracy, correct_predict, predictions

    def predict(self, x_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.dropout_keep_prob: 1.0
        }
        predictions = self.sess.run([self.predictions], feed_dict)
        return predictions

    def get_score(self, x_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.dropout_keep_prob: 1.0
        }
        scores = self.sess.run([self.scores], feed_dict)
        return scores

    def save_model(self, model_path, global_step):
        save_path = self.saver.save(self.sess, model_path, global_step=global_step)
        print("model saved in file: %s" % save_path)

    def restore(self, restore_path):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        self.saver.restore(self.sess, restore_path)
        print("model restored")