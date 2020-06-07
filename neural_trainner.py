from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten

stopwords = set(stopwords.words('english') + list(punctuation))

MAX_SEQUENCE_LENGTH = 750


def check_length(filedir: str):  # powiedzmy 750
    l = []
    for f in os.listdir(filedir):
        s = ""
        with open(f"{filedir}/{f}", 'r', encoding="cp1250") as file:
            s = file.read()
        tok = word_tokenize(s)
        l.append(np.array(tok, dtype=str).size)
    print(np.average(l))
    print(np.std(l))


def make_dict(size: int, filedir: str = "data/lyrics") -> dict:
    d = dict()
    for genre in os.listdir(filedir):
        for f in os.listdir(f"{filedir}/{genre}"):
            s = ""
            with open(f"{filedir}/{genre}/{f}", 'r', encoding='cp1250') as file:
                s = file.read()

            tok = word_tokenize(s.lower())  # zakładam, że zawsze jest angielski
            for t in tok:
                # if t not in stopwords:  # glove ma dla nich embeddingi więc zobaczymy jak działa bez, a najwyżej się będzie zamieniać
                if t in d.keys():
                    d[t] += 1
                else:
                    d[t] = 1

    # print(d, len(d))
    d = sorted(d.items(), key=lambda x: x[1],
               reverse=True)  # rozważyć co lepiej odcinać, najczęściej występujące czy najmniej
    if len(d) < size:
        sizen = len(d)
    else:
        sizen = size
    d = d[:sizen]

    dictionary = dict()
    index = 1
    for i in d:
        dictionary[i[0]] = index
        index += 1

    with open(f"data/dictionaries/dict{size}.pickle", 'wb') as file:
        pickle.dump(dictionary, file)

    return dictionary


def convert_lyrics(dic: dict = None):
    if dic is None:
        with open("data/dictionaries/dict10000.pickle", 'rb') as file:
            dic = pickle.load(file)

    for genre in os.listdir("data/lyrics"):
        ind = []
        for f in os.listdir(f"data/lyrics/{genre}"):
            s = ""
            with open(f"data/lyrics/{genre}/{f}", 'r', encoding="cp1250") as file:
                s = file.read()
        tok = word_tokenize(s.lower())
        indexes = []
        for t in tok:
            if t in dic.keys():
                indexes.append(dic[t])
            ind.append(indexes)
        with open(f"data/dictionaries/lyrics/{genre}.pickle", 'wb') as file:
            pickle.dump(ind, file)
        print(ind)


EMBEDDING_DIM = 100


def make_embeddings(dictionary: dict, size: int):
    embedding = dict()
    with open(f"/home/er713/glove/glove.6B.{EMBEDDING_DIM}d.txt", 'r') as glove:
        for line in glove:
            val = line.split()
            embedding[val[0]] = np.asarray(val[1:], dtype='float32')

    emb_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    for word, i in dictionary.items():
        if word in embedding.keys():
            emb_matrix[i] = embedding[word]

    with open(f"data/dictionaries/dict{size}emb{EMBEDDING_DIM}.pickle", 'wb') as file:
        pickle.dump(emb_matrix, file)

    return emb_matrix


def prepare_data_and_train():
    pass


def train_network(train_x, train_y, test_x, test_y, embedding_matrix, train_embedding: bool):
    model = Sequential()
    Embedding()
    model.add(Embedding(len(embedding_matrix) + 1, EMBEDDING_DIM, embeddings_initializer=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH, trainable=train_embedding))
    model.add(Conv1D(250, 5, activation='relu'))
    model.add(MaxPooling1D(25))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # musi być liczbą opcji

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=2, batch_size=128)


if __name__ == "__main__":
    # check_length("data/lyrics/pop")
    # print(make_dict(10000))
    # convert_lyrics()
    SIZE = 100
    d = make_dict(SIZE)
    # emb = make_embeddings(d, SIZE)
    # print(emb)
