import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string

batch_size = 50

# Read the data into a list of strings.
def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    return

def read_data():
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
    for f in file_list:
        with open(f, "r", encoding="utf-8") as openf:
            try:
                s = openf.read()
            except:
                print(f);
                raise;
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            data.append(no_punct.split())

    return data

def load_data(glove_dict):
    """ Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    extract_data('reviews.tar.gz') # unzip
    vocabulary = read_data()
    data = np.zeros((len(vocabulary), 40))

    #do things to data here

    for i, review in enumerate(vocabulary):
        j = 0;
        for word in review:
            index = glove_dict.get(word, None);
            if index == None:
                continue;

            data[i][j] = index;
            j += 1;

            if j >= 40:
                break

    del vocabulary  # Hint to reduce memory.
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    word_index_dict = {'UNK' : 0};
    embeddings = [[0] * 50];

    #with open("glove.6B.50d.txt",'r',encoding="utf-8") as data: 
    with open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8") as data:
        for line in data:
            words = line.split();
            word_index_dict[words[0]] = len(embeddings);
            embeddings.append([float(i) for i in words[1:]]);

    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""

    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    embeddings = tf.convert_to_tensor(glove_embeddings_arr); #[vocab, 50]
    input_data = tf.placeholder(shape=[batch_size, 40], name="input_data");
    word_embeddings = tf.nn.embedding_lookup(embeddings, input_data); #[batch_size, 40, 50]

    state_size = 512;
    rnn = rnn.BasicLSTMCell(state_size);
    outputs, states = rnn.static_rnn(rnn, word_embeddings, dtype=tf.float32);

    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name="labels")


    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
