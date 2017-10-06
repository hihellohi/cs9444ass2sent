import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string.punctuation

batch_size = 50

def check_file(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(filename))
    return filename


# Read the data into a list of strings.
def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    return

def read_data():
    print("READING DATA")
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
    print("Parsing %s files" % len(file_list))
    for f in file_list:
        with open(f, "r") as openf:
            try:
                s = openf.read()
            except:
                print(f);
                raise;
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            data.extend(no_punct.split())

    print(data[:5])
    return data

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def get_dataset(vocabulary_size):
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed data, to reparse, delete 'data.npy'")
        data = np.load("data.npy")
        count = np.load("count.npy")
        dictionary = np.load("Word2Idx.npy").item()
        reverse_dictionary = np.load("Idx2Word.npy").item()
    else:
        filename = check_file('reviews.tar.gz', 14839260)
        extract_data(filename) # unzip
        vocabulary = read_data()
        print('Data size', len(vocabulary))
        # Step 2: Build the dictionary and replace rare words with UNK token.
        data, count, dictionary, reverse_dictionary =\
            build_dataset(vocabulary, vocabulary_size)

        np.save("data", data)
        np.save("count", count)
        np.save("Idx2Word", reverse_dictionary)
        np.save("Word2Idx", dictionary)
        del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
	data, count, dictionary, reverse_dictionary = get_dataset(vocabulary_size)
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
    #data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
	translator = str.maketrans('','',string.punctuation);
	for line in data:
		words = line.lower().translate(translator).split();

	data.close();
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

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
