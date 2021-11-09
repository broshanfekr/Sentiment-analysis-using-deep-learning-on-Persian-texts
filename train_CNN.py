import numpy as np
import os
import time
import data_helpers
import parsivar
from persian_CNN import TextCNN
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import timeit

parser = data_helpers.create_parser()
FLAGS = parser.parse_args()

print("\nParameters:")
for arg in vars(FLAGS):
    print("{} : {}".format(arg, getattr(FLAGS, arg)))
print("")


def read_from_dataset(dataset_path, word2vec_model_path, normalizer, tokenizer, n_classes, max_seq_len_cutoff):
    print("Loading data...")
    x, y, sequence_length, word2vec_vocab, un_known = data_helpers.load_data(dataset_path,
                                                                             word2vec_model_path,
                                                                             normalizer,
                                                                             tokenizer,
                                                                             n_classes,
                                                                             True,
                                                                             max_seq_len_cutoff)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=12)
    X_train_balanced, Y_train_balanced = data_helpers.balance_dataset(X_train, Y_train, n_classes=n_classes)

    x_train, x_dev = X_train_balanced, X_test
    y_train, y_dev = Y_train_balanced, Y_test

    if FLAGS.is_evaluation is True:
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        FLAGS.evaluate_every=1

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("max_seq_len is: ", sequence_length)

    return x_train, x_dev, y_train, y_dev, sequence_length, word2vec_vocab, un_known


def my_main():
    my_normalizer = parsivar.Normalizer()
    my_tokenizer = parsivar.Tokenizer()

    x_train, x_dev, y_train, y_dev, seq_max_len, word2vec_vocab, un_known = read_from_dataset(FLAGS.input_dataset_path,
                                                                                              FLAGS.word2vec_model_path, my_normalizer,
                                                                                              my_tokenizer,
                                                                                              FLAGS.n_classes,
                                                                                              FLAGS.max_seq_len_cutoff)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=seq_max_len,
                num_classes=2,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=40)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch, epoch_num):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, correct_predict, weigth_norm, batch_predictions = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.correct_predictions, cnn.weigth_norm, cnn.predictions],
                    feed_dict)

                cm_normalized, prec, rec, f_measure = data_helpers.model_evaluation(model_results=batch_predictions,
                                                                                    true_results=np.argmax(y_batch, axis=1))

                print("epoch: {:g}, iteration: {:g}, weigth_norm: {:.6f},  loss: {:.4f}, "
                      "acc: {:.4f}, F-score: {:.5f}".format(epoch_num, step, weigth_norm, loss, accuracy, f_measure))

            def dev_step(x_batch, y_batch, test_set="dev"):
                """
                Evaluates model on a dev set
                """
                dev_baches = data_helpers.batch_iter(list(zip(x_batch, y_batch)),
                                                     batch_size=FLAGS.batch_size,
                                                     seq_length=seq_max_len,
                                                     emmbedding_size=FLAGS.embedding_dim,
                                                     word2vec_wv=word2vec_vocab,
                                                     un_known=un_known,
                                                     is_shuffle=False)

                total_loss = 0
                total_acc = 0
                index = 0
                total_correct_predictions = 0
                total_dev_data = 0

                all_predictions = []
                y_test = []
                for batch in dev_baches:
                    if (len(batch[0]) == 0):
                        continue
                    x_batch, y_batch = zip(*batch[0])
                    total_dev_data += len(x_batch)

                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }

                    step, loss, accuracy, correct_predict, batch_predictions = sess.run(
                        [global_step, cnn.loss, cnn.accuracy, cnn.correct_predictions, cnn.predictions],
                        feed_dict)
                    y_test = np.concatenate([y_test, np.argmax(y_batch, axis=1)])
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

                    cm_normalized, prec, rec, f_measure = data_helpers.model_evaluation(model_results=batch_predictions,
                                                                                        true_results=np.argmax(y_batch, axis=1))

                    print("on {}, test index: {:g}, Minibatch Loss: {:.6f}, acc: {:.5f}, "
                          "F-score: {:.5f}".format(test_set, index, loss, accuracy, f_measure))

                    total_loss += loss
                    total_acc += accuracy
                    index += 1
                    total_correct_predictions += np.sum(correct_predict)

                print("#################################################################\n")
                avg_loss = total_loss / (index)
                avg_acc = total_acc / (index)
                real_acc = (total_correct_predictions*1.0) / (total_dev_data)

                print("on {}, avarage_Loss: {:.6f}, avarage_acc: {:.5f}, "
                      "real_acc: {:.5f}\n".format(test_set, avg_loss, avg_acc, real_acc))

                cm_normalized, prec, rec, f_measure = data_helpers.model_evaluation(model_results=all_predictions,
                                                                                    true_results=y_test)
                print(cm_normalized)
                print("prec is: ", prec, "      rec is: ", rec, "    f_measure is: ", f_measure)
                print("---------------------------------------------------------------------------------------")
                # Print accuracy
                correct_predictions = float(sum(all_predictions == y_test))
                print("Total number of test examples: {}".format(len(y_test)))
                print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
                return avg_loss, real_acc

            train_loss_list = []
            train_acc_list = []
            test_loss_list = []
            test_acc_list = []

            avg_loss, real_acc = dev_step(x_dev, y_dev)

            for epoch in range(FLAGS.num_epochs):
                batches = data_helpers.batch_iter(list(zip(x_train, y_train)),
                                                  batch_size=FLAGS.batch_size,
                                                  seq_length=seq_max_len,
                                                  emmbedding_size=FLAGS.embedding_dim,
                                                  word2vec_wv=word2vec_vocab, un_known=un_known)
                # Training loop. For each batch...
                for batch in batches:
                    if len(batch[0]) == 0:
                        continue
                    x_batch, y_batch = zip(*batch[0])
                    current_step = tf.train.global_step(sess, global_step)
                    train_step(x_batch, y_batch, epoch)

                if (epoch+1) % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=(epoch+1))
                    print("Saved model checkpoint to {}\n".format(path))

                if ((epoch+1) % FLAGS.evaluate_every) == 0:
                    print("testing on dev set: ")
                    avg_loss, real_acc = dev_step(x_dev, y_dev)

                    test_loss_list.append(avg_loss)
                    test_acc_list.append(real_acc)

            path = saver.save(sess, checkpoint_prefix, global_step=FLAGS.num_epochs)
            print("Saved model checkpoint to {}\n".format(path))
            print("Optimization Finished!")
            print("\nEvaluation:")
            dev_step(x_dev, y_dev)
            print("")

            return train_acc_list, train_loss_list, test_acc_list, test_loss_list


if __name__ == "__main__":
    if not os.path.isfile(FLAGS.input_dataset_path):
        os.system("gdown --id 10b55IBvYO56cjLvfBuxIbe5vs-bsIlB0")

    start = timeit.default_timer()
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = my_main()
    stop = timeit.default_timer()

    spent_time = int(stop - start)
    sec = spent_time % 60
    spent_time = int(spent_time / 60)
    minute = spent_time % 60
    spent_time = int(spent_time / 60)
    hours = spent_time
    print("h: ", hours, "  minutes: ", minute, "  secunds: ", sec)
