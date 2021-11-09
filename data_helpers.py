import numpy as np
from gensim.models import Word2Vec
import copy
import random
import argparse
from collections import Counter
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import parsivar


# creating a parser for arguments
def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dataset_path', '-input_dataset_path', type=str, default="digikala_dataset.txt",
        metavar='PATH',
        help="Path of the dataset"
    )
    parser.add_argument(
        '--word2vec_model_path', '-word2vec_model_path', type=str, default="word2vec.model",
        metavar='PATH',
        help="the path of trained word2vec model."
    )
    parser.add_argument(
        '--is_separated', '-separate', type=bool, default=False,
        help="does the dataset have official train/test split or not?"
    )

    parser.add_argument(
        '--max_seq_len_cutoff', '-max_seq_len_cutoff', type=int, default=1500,
        help="the max len of a sentence."
    )
    parser.add_argument(
        '--learning_rate', '-learning_rate', type=float, default=0.001,
        help="learning rate parameter"
    )
    parser.add_argument(
        '--n_classes', '-n_classes', type=int, default=2,
        help="number of classes in classification problem"
    )
    parser.add_argument(
        '--batch_size', '-batch_size', type=int, default=100,
        help="size of the batch in training step"
    )
    parser.add_argument(
        '--num_epochs', '-num_epochs', type=int, default=50,
        help="defines the number of training epochs."
    )
    parser.add_argument(
        '--filter_sizes', '-filter_sizes', type=str, default="3,4,5",
        help="Comma-separated filter sizes (default: '3,4,5')"
    )
    parser.add_argument(
        '--num_filters', '-num_filters', type=int, default=150,
        help="Number of filters per filter size (default: 150)"
    )
    parser.add_argument(
        '--embedding_dim', '-embedding_dim', type=int, default=150,
        help="Dimensionality of word embedding"
    )

    parser.add_argument(
        '--n_hidden_attention', '-n_hidden_attention', type=int, default=100,
        help="Dimensionality of attention hidden layer"
    )
    parser.add_argument(
        '--evaluate_every', '-evaluate_every', type=int, default=3,
        help="Evaluate model on dev set after this many steps"
    )
    parser.add_argument(
        '--checkpoint_every', '-checkpoint_every', type=int, default=3,
        help="Save model after this many steps"
    )
    parser.add_argument(
        '--l2_reg_lambda', '-l2_reg_lambda', type=float, default=0.1,
        help="L2 regularizaion lambda"
    )

    parser.add_argument(
        '--dropout_keep_prob', '-dropout_keep_prob', type=float, default=0.5,
        help="Dropout keep probability (default: 0.5)"
    )
    parser.add_argument(
        '--checkpoint_dir', '-checkpoint_dir', type=str, default="runs/checkpoints",
        help="Checkpoint directory from training run"
    )
    parser.add_argument(
        '--allow_soft_placement', '-allow_soft_placement', type=bool, default=True,
        help="Allow device soft device placement"
    )
    parser.add_argument(
        '--log_device_placement', "-log_device_placement", type=bool, default=False,
        help="Log placement of ops on devices"
    )
    parser.add_argument(
        '--is_load_model', '-is_load_model', type=bool, default=False,
        help="do we want to load previes model?"
    )
    parser.add_argument(
        '--is_evaluation', '-is_evaluation', type=bool, default=False,
        help="do we want to use train set as evaluation?"
    )

    return parser


def balance_dataset(x, y, n_classes):
    label_list = []
    class_dict = {i: [] for i in range(n_classes)}
    for i, label in enumerate(y):
        index = list(label).index(1)
        label_list.append(index)
        class_dict[index].append(i)

    each_label_count = Counter(label_list)
    each_label_count = [each_label_count[key] for key in each_label_count]
    max_label = max(each_label_count)

    extend_list = []
    extend_label = []
    for key in class_dict:
        class_list = class_dict[key]
        class_len = len(class_list)
        residual = max_label-class_len

        for i in range(residual):
            sample_idx = class_list[i % class_len]
            extend_list.append(x[sample_idx])
            extend_label.append(y[sample_idx])

    extend_label = np.asarray(extend_label)
    x.extend(extend_list)
    y = np.concatenate([y, extend_label], axis=0)
    return x, y


def train_word2vec_model(dataset_path, save_path="word2vec.model"):
    with open(dataset_path, "r", encoding="utf-8") as infile:
        content = infile.readlines()

    x_text = []
    for line in content:
        line = line.strip().split("\t")
        text = line[1].strip()
        if len(text) == 0:
            continue
        x_text.append(text)

    x_text = clean_str(x_text, normalizer=parsivar.Normalizer(), tokenizer=parsivar.Tokenizer(),
                       is_cut=False, max_seq_len_cutoff=None)

    model = Word2Vec(sentences=x_text, vector_size=150, window=5, min_count=5, sg=1, epochs=20, workers=6)
    model.save(save_path)
    return model


def load_w2v_model(name):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    return Word2Vec.load(dir_path + name)


def clean_str(review_docs, normalizer, tokenizer, is_cut, max_seq_len_cutoff):
    output_docs = []

    for i, string in enumerate(review_docs):
        if i % 5000 == 0:
            print("cleaning text {:d} of {:d}".format(i, len(review_docs)))
        words = tokenizer.tokenize_words(normalizer.normalize(string))

        if is_cut and len(words) > max_seq_len_cutoff:
            words = words[:max_seq_len_cutoff]

        for index, w in enumerate(words):
            if w.replace('.', '', 1).isdigit():
                words[index] = '<number>'
        output_docs.append(words)
    return output_docs


def build_input_data_from_word2vec(sentence, word2vec_wv, un_known):
    """
    Maps sentencs and labels to vectors based on a word2vec model.
    """
    x_data = []
    for word in sentence:
        try:
            word_vector = word2vec_wv[word]
        except:
            word_vector = un_known
        x_data.append(word_vector)
    x_data = np.asarray(x_data)
    return x_data


def load_data(dataset_path, word2vec_model_path, normalizer, tokenizer, n_class, is_cut, max_seq_len_cutoff):
    """
    Loads and preprocessed data from dataset file.
    """
    with open(dataset_path, "r", encoding="utf-8") as infile:
        content = infile.readlines()

    x_text = []
    label_list = []
    for line in content:
        line = line.strip().split("\t")
        if line[0] == '0':
            continue

        label = int(line[0])
        if label == -1:
            label = 0
        text = line[1].strip()
        if len(text) == 0:
            continue

        x_text.append(text)
        tmp_lable = np.zeros(n_class)
        tmp_lable[label] = 1
        label_list.append(tmp_lable)

    x_text = clean_str(x_text, normalizer=normalizer, tokenizer=tokenizer, is_cut=is_cut,
                       max_seq_len_cutoff=max_seq_len_cutoff)

    y = np.asarray(label_list)
    sequence_length = max(len(x) for x in x_text)

    if os.path.isfile(word2vec_model_path):
        word2vec_model = load_w2v_model(word2vec_model_path)
        word2vec_vocab = word2vec_model.wv
    else:
        word2vec_model = train_word2vec_model(dataset_path, word2vec_model_path)
        word2vec_vocab = word2vec_model.wv

    print("word2vec len is: ", word2vec_vocab.vector_size)
    un_known = np.random.uniform(low=-0.25, high=0.25, size=word2vec_vocab.vector_size)

    return [x_text, y, sequence_length, word2vec_vocab, un_known]


def build_test_data_matrix(data, seq_length, emmbedding_size, word2vec_wv,
                           un_known, normalizer, tokenizer):
    data = clean_str(data, normalizer=normalizer, tokenizer=tokenizer, is_cut=True,
                     max_seq_len_cutoff=seq_length)
    tmp_data = copy.deepcopy(data)
    tmp_docs = []
    for x in tmp_data:
        doc_vector = build_input_data_from_word2vec(x, word2vec_wv, un_known)
        if len(doc_vector) < seq_length:
            num_padding = seq_length - len(x)
            x_bar = np.zeros([num_padding, emmbedding_size])
            doc_vector = np.concatenate([doc_vector, x_bar], axis=0)
        tmp_docs.append(doc_vector)

    tmp_docs = np.asarray(tmp_docs)
    return tmp_docs


def batch_iter(data, batch_size, seq_length, emmbedding_size, word2vec_wv, un_known, is_shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    # Shuffle the data at each epoch
    if is_shuffle:
        random.shuffle(data)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        tmp_data = copy.deepcopy(data[start_index:end_index])

        batch_seq_len = []
        tmp_docs = []
        tmp_labels = []
        for x in tmp_data:
            batch_seq_len.append(len(x[0]))
            doc_vector = build_input_data_from_word2vec(x[0], word2vec_wv, un_known)
            if len(doc_vector) < seq_length:
                num_padding = seq_length - len(x[0])
                x_bar = np.zeros([num_padding, emmbedding_size])
                doc_vector = np.concatenate([doc_vector, x_bar], axis=0)
            tmp_docs.append(doc_vector)
            tmp_labels.append(x[1])

        tmp_docs = np.asarray(tmp_docs)
        tmp_labels = np.asarray(tmp_labels)
        tmp_data = list(zip(tmp_docs, tmp_labels))
        yield [tmp_data, batch_seq_len]


def my_confusion_matrix(model_results, true_results):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, element in enumerate(model_results):
        if element == true_results[i]:  # true
            if element == 1:  # negative
                tn += 1
            else:  # positive
                tp += 1
        else:  # false
            if element == 1:  # negative
                fn += 1
            else:  # positive
                fp += 1
    return tp, tn, fp, fn


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def model_evaluation(model_results, true_results):
    cm = confusion_matrix(true_results, model_results)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    tp, tn, fp, fn = my_confusion_matrix(model_results=model_results, true_results=true_results)
    if(tp+fp) != 0:
        prec = float(tp)/(tp+fp)
    else:
        prec = 0
    if(tp+fn) != 0:
        rec = float(tp)/(tp+fn)
    else:
        rec = 0
    if(prec + rec) != 0:
        f_measure = float(2*prec*rec)/(prec + rec)
    else:
        f_measure = 0

    return cm, prec, rec, f_measure