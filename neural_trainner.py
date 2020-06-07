from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split

stopwords = set(stopwords.words('english') + list(punctuation))

MAX_SEQUENCE_LENGTH = 650


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
        # count = 0
        ind = []
        for f in os.listdir(f"data/lyrics/{genre}"):
            s = ""
            with open(f"data/lyrics/{genre}/{f}", 'r', encoding="cp1250") as file:
                s = file.read()
                if s is None or s == "":
                    # count += 1
                    continue
            tok = word_tokenize(s.lower())
            indexes = []
            for t in tok:
                if t in dic.keys():
                    indexes.append(dic[t])
            ind.append(indexes)
        # print(ind[0], len(ind[0]))
        # print(len(os.listdir(f"data/lyrics/{genre}")), len(ind), np.sum([len(q) for q in ind]))
        # print(count)
        with open(f"data/dictionaries/lyrics/{genre}.pickle", 'wb') as file:
            pickle.dump(ind, file)


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
    train_x, train_y, test_x, test_y = [], [], [], []
    answer_key = dict()
    quant = len(os.listdir("data/dictionaries/lyrics"))
    for i, f in enumerate(os.listdir("data/dictionaries/lyrics")):
        answer_key[f[:-7]] = i
        split = []
        with open(f"data/dictionaries/lyrics/{f}", 'rb') as file:
            split = pickle.load(file)
        # print(len(split))
        x, t, _, _ = train_test_split(split, np.zeros(len(split)), test_size=0.3, shuffle=True)
        y = np.zeros(quant, dtype=int)
        y[i] = 1
        for j in x:
            train_x.append(j)
            train_y.append(y)
        for j in t:
            test_x.append(j)
            test_y.append(y)
    train_x = pad_sequences(train_x, MAX_SEQUENCE_LENGTH)
    test_x = pad_sequences(test_x, MAX_SEQUENCE_LENGTH)

    # return train_x, train_y, test_x, test_y, answer_key
    with open("data/dictionaries/dict10000emb100.pickle", 'rb') as file:
        emb_matrix = pickle.load(file)
    return train_network(train_x, train_y, test_x, test_y, emb_matrix, answer_key), answer_key, test_x, test_y


def train_network(train_x, train_y, test_x, test_y, embedding_matrix, answer_key):
    model = Sequential()
    model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    model.add(Conv1D(250, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(250, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(250, 3, activation='relu'))
    model.add(MaxPooling1D(20))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(answer_key), activation='softmax'))  # musi być liczbą opcji

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # print(np.asarray(test_x)[0], train_y[0], "\n", test_x[0], test_y[0])
    model.fit(np.asarray(train_x), np.asarray(train_y), validation_data=(np.asarray(test_x), np.asarray(test_y)),
              epochs=4, batch_size=256)

    with open("data/networks/networkTest.pickle", 'wb') as file:
        pickle.dump(model, file)

    return model


if __name__ == "__main__":
    # check_length("data/lyrics/pop")
    # print(make_dict(10000))
    # convert_lyrics()
    # SIZE = 10000
    # d = make_dict(SIZE)
    # print(d, len(d))
    # emb = make_embeddings(d, SIZE)
    # print(emb, len(emb))
    # with open("data/dictionaries/dict10000emb100.pickle", 'rb') as file:
    #     d = pickle.load(file)
    #     print(len(d))
    model, answer_key, test_x, test_y = prepare_data_and_train()
    # wyn = model.predict(test_x, batch_size=128)
    wyn2 = model.predict_classes(test_x, batch_size=128)
    count = 0
    for w2, r in zip(wyn2, test_y):
        # print(w1, w2, r)
        if r[w2] == 1:
            count += 1
    print(count / len(test_y) * 100, '%')
