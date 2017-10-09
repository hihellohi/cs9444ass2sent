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
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        data = np.load("data.npy")
        return data;

    extract_data('reviews.tar.gz') # unzip
    vocabulary = read_data()
    data = np.zeros((len(vocabulary), 40))

    #dict used for O(1) lookup
    useless_words = {
            'the' : 1,
            'be' : 1,
            'to' : 1,
            'of' : 1,
            'and' : 1,
            'a' : 1,
            'in': 1,
            'that' : 1,
            'have' : 1,
            'i' : 1,
            'it' : 1,
            'for' : 1}

    for i, review in enumerate(vocabulary):
        j = 0;
        for word in review:
            if word in useless_words:
                continue;

            index = glove_dict.get(word, None);
            if index == None:
                continue;

            data[i][j] = index;
            j += 1;

            if j >= 40:
                break

    del vocabulary  # Hint to reduce memory.

    np.save("data", data)
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

    with open("glove.6B.50d.txt",'r',encoding="utf-8") as data: 
    #with open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8") as data:
        for line in data:
            words = line.split();
            word_index_dict[words[0]] = len(embeddings);
            embeddings.append([float(i) for i in words[1:]]);

    return embeddings, word_index_dict

def new_lstm(state_size, dropout_prob):
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(state_size), state_keep_prob=dropout_prob);

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
    state_size = 64;
    learning_rate = 0.0001;
    momentum = 0.9;
    num_layers = 2;

    dropout_keep_prob = tf.placeholder_with_default(0.5, shape=())

    embeddings = tf.convert_to_tensor(glove_embeddings_arr); #[vocab, 50]
    input_data = tf.placeholder(shape=[batch_size, 40], name="input_data", dtype=tf.int32);
    word_embeddings = tf.nn.embedding_lookup(embeddings, tf.transpose(input_data)); #[40, batch_size, 50]
    iterable = tf.split(tf.reshape(word_embeddings, [40*batch_size, 50]), 40, 0);

    fwd = new_lstm(state_size, dropout_keep_prob);
    back = new_lstm(state_size, dropout_keep_prob);
    outputs, state1, state2 = tf.contrib.rnn.static_bidirectional_rnn(fwd, back, iterable, dtype=tf.float32);

    rnn = tf.contrib.rnn.MultiRNNCell( [new_lstm(state_size, dropout_keep_prob) for _ in range(num_layers)]);
    outputs, states = tf.nn.static_rnn(rnn, outputs, dtype=tf.float32); #outputs[-1] has shape [batch_size, state_size] 

    output_weights = tf.Variable(tf.random_normal([state_size, 2]));
    output_bias = tf.Variable(tf.random_normal([2]));
    logits = tf.matmul(outputs[-1], output_weights) + output_bias; #[batch_size, 2]

    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name="labels")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name="loss");
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss);

    correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
