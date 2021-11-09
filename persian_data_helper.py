import numpy as np
import itertools
from collections import Counter
import tarfile
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import gensim
import copy
import os


def Load_Doc2Vec_Model(name='./myIMDB_model.d2v'):
    return Doc2Vec.load(name)

def Load_Word2Vec_Model(name = "./myImDB_model.d2v"):
    return Word2Vec.load(name)

def clean_str(review_docs, max_seq_len_cutoff):
    output_docs = []
    for string in review_docs:
        tmp = normalizer.normalize(string)
        #print(tmp)
        # print(sent_tokenize(tmp))
        words = word_tokenize(tmp)

        if(len(words) >  max_seq_len_cutoff):
            words = words[:max_seq_len_cutoff]
        #words = gensim.utils.to_unicode(string).split()
        if(len(words) != 0):
            output_docs.append(words)


    return output_docs


def Load_Digikala_Data_and_Label(dataset_file_name, max_seq_len_cutoff, is_remove_stopwords = False):
    with open(dataset_file_name, "r", encoding="utf-8") as infile:
        content = infile.readlines()
        x_text = []
        label_list = []
        for line in content:
            line = line.strip().split("\t")
            if line[0] == '0':
                continue

            label = int(line[0])
            text = line[1].strip()
            if len(text) == 0:
                continue

            x_text.append(text)
            tmp_lable = np.zeros(n_classes)
            tmp_lable[label] = 1
            label_list.append(tmp_lable)

        x_text = clean_str(x_text, normalizer=normalizer, tokenizer=tokenizer, is_cut= is_cutoff_required, max_seq_len_cutoff=max_seq_len_cutoff)
        x_train.extend(x_text)
        label_list = np.asarray(label_list)
        y_train = np.concatenate([y_train, label_list], axis=0)
        sequence_length = max(len(x) for x in x_train)
        return x_train, y_train, sequence_length

    for member in os.listdir(dataset_file_name):
        if(member[-1] == "~"):
            continue
        file_name = member.split("_")
        data_label = file_name[-1].split(".")[0]
        israte = file_name[-2]
        if(israte == 'rates'):
            continue
        f = open(dataset_file_name + '/' + member, "r")
        content = f.readlines()
        f.close()
        if(data_label == "pos"):
            positive_examples = content
            print('posetive data parsed from data set')
        elif(data_label == "neg"):
            negative_examples = content
            print("negative data parsed from data set")
        else:
            unsup_examples = content
            print("unsup data parsed from data set")

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    #unsup_examples = [s.strip() for s in unsup_examples]
    # Split by words
    positive_examples = clean_str(positive_examples, max_seq_len_cutoff)
    negative_examples = clean_str(negative_examples, max_seq_len_cutoff)
    #unsup_examples = clean_str(unsup_examples)

    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]

def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_sentiment_document(sentences, labels):
    from collections import namedtuple
    SentimentDocument = namedtuple('SentimentDocument', 'words tags sentiment')
    alldocs = []
    for line_no, line in enumerate(sentences):
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        if(line_no < len(labels)):
            sentiment = labels[line_no][1]
        else:
            sentiment = None
        alldocs.append(SentimentDocument(line, tags, sentiment))
    return alldocs

def load_pretrained_data(dataset_path, word2vec_model_path, n_class=2, max_seq_len_cutoff=1350):
    # Load and preprocess data
    sentences, labels = Load_Digikala_Data_and_Label(dataset_path, max_seq_len_cutoff)
    sequence_length = max(len(x) for x in sentences)

    vocabulary, vocabulary_inv = build_vocab(sentences)

    word2vec_Model = Load_Word2Vec_Model(name=word2vec_model_path)
    word2vec_vocab = word2vec_Model.vocab
    word2vec_vec = word2vec_Model.syn0

    print("word2vec len is: ", len(word2vec_vec))
    tmp = word2vec_vocab["خوب"]
    tmp1 = copy.deepcopy(tmp)
    unknown_word_vector = np.random.uniform(low=-0.25, high=0.25, size=(1, word2vec_vec.shape[1]))
    word2vec_vec = np.append(word2vec_vec, unknown_word_vector, axis=0)
    tmp1.index = len(word2vec_vec)-1
    word2vec_vocab['<un_known>'] = tmp1

    return [sentences, labels, sequence_length ,vocabulary, vocabulary_inv, word2vec_vocab, word2vec_vec]

def build_input_data_from_word2vec(sentence, word2vec_vocab, word2vec_vec):
    """
    Maps sentenc and vectors based on a word2vec model.
    """
    X_data = []
    for word in sentence:
        try:
            word2vec_index = word2vec_vocab[word].index
            word_vector = word2vec_vec[word2vec_index]
        except:
            word2vec_index = word2vec_vocab['<un_known>'].index
            word_vector = word2vec_vec[word2vec_index]
            #word_vector = np.random.uniform(low=-0.25, high=0.25, size=word2vec_vec.shape[1])
        X_data.append(word_vector)
    X_data = np.asarray(X_data)
    return X_data

def batch_iter(data, batch_size, seq_length, emmbedding_size,word2vec_vocab, word2vec_vec, is_shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    import random
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
            doc_vector = build_input_data_from_word2vec(x[0], word2vec_vocab, word2vec_vec)
            if(len(doc_vector) < seq_length):
                num_padding = seq_length - len(x[0])
                x_bar = np.zeros([num_padding, emmbedding_size])
                doc_vector = np.concatenate([doc_vector, x_bar], axis=0)
            tmp_docs.append(doc_vector)
            tmp_labels.append(x[1])
        tmp_docs = np.asarray(tmp_docs)
        tmp_labels = np.asarray(tmp_labels)

        tmp_data = list(zip(tmp_docs, tmp_labels))
        yield [tmp_data, batch_seq_len]