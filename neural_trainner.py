from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, save_model
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten, AveragePooling1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

stopwords = set(stopwords.words('english') + list(punctuation))


def check_length(filedir: str):
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
                if t not in stopwords:  # glove ma dla nich embeddingi więc zobaczymy jak działa bez, a najwyżej się będzie zamieniać
                    if t in d.keys():
                        d[t] += 1
                    else:
                        d[t] = 1

    # print(d, len(d))
    d = sorted(d.items(), key=lambda x: x[1],
               reverse=True)  # rozważyć co lepiej odcinać, najczęściej występujące czy najmniej
    print(len(d))
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
        with open(f"data/dictionaries/dict{DICTIONARY_DIM}.pickle", 'rb') as file:
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
    length = []
    train_x, train_y, test_x, test_y = [], [], [], []
    answer_key = []
    quant = len(os.listdir("data/dictionaries/lyrics"))
    for i, f in enumerate(os.listdir("data/dictionaries/lyrics")):
        answer_key.append(f[:-7])
        split = []
        with open(f"data/dictionaries/lyrics/{f}", 'rb') as file:
            split = pickle.load(file)
        # print(len(split))
        [length.append(len(q)) for q in split]
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

    print(np.average(length), np.std(length))
    # return train_x, train_y, test_x, test_y, answer_key
    with open(f"data/dictionaries/dict{DICTIONARY_DIM}emb{EMBEDDING_DIM}.pickle", 'rb') as file:
        emb_matrix = pickle.load(file)
    return train_network(train_x, train_y, test_x, test_y, emb_matrix, answer_key), answer_key, test_x, test_y


def train_network(train_x, train_y, test_x, test_y, embedding_matrix, answer_key):
    model = Sequential()
    model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    # model.add(Conv1D(EMBEDDING_DIM, 3, activation='relu'))
    # model.add(AveragePooling1D(3))
    # model.add(Conv1D(EMBEDDING_DIM, 3, activation='relu'))
    # model.add(AveragePooling1D(3))
    model.add(Conv1D(EMBEDDING_DIM, 3, activation='relu'))  # padding="same",
    # model.add(AveragePooling1D(20))
    model.add(GlobalAveragePooling1D())
    # model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(answer_key), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # print(np.asarray(test_x)[0], train_y[0], "\n", test_x[0], test_y[0])
    acc, val_acc, epoch = 0., 0., 0
    while val_acc <= 0.77 and (acc <= val_acc + 0.03 or epoch < 2):
        history = model.fit(np.asarray(train_x), np.asarray(train_y),
                            validation_data=(np.asarray(test_x), np.asarray(test_y)),
                            epochs=1, batch_size=32)
        # print(history.history)
        acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]
        epoch += 1
        print(acc, val_acc, epoch)
    # model.fit(np.asarray(train_x), np.asarray(train_y),
    #           validation_data=(np.asarray(test_x), np.asarray(test_y)),
    #           epochs=3, batch_size=32)

    with open("data/networks/networkTest.pickle", 'wb') as file:
        pickle.dump(model, file)

    return model


def draw_confusion_matrix(conf, id_to_genre):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.matshow(conf, alpha=0.3)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(x=j, y=i, s=conf[i, j], va='center', ha='center')
    plt.xticks(list(range(4)), id_to_genre)
    plt.yticks(list(range(4)), id_to_genre)
    plt.xlabel('Przewidziana etykieta')
    plt.ylabel('Rzewczywista etykieta')
    plt.savefig('confusion_matrix_net.png')


EMBEDDING_DIM = 50
DICTIONARY_DIM = 5000

MAX_SEQUENCE_LENGTH = 750

if __name__ == "__main__":

    # d = make_dict(DICTIONARY_DIM)
    # emb = make_embeddings(d, DICTIONARY_DIM)
    #
    # convert_lyrics()
    #
    # model, answer_key, test_x, test_y = prepare_data_and_train()
    # wyn = model.predict(test_x, batch_size=128)
    # wyn2 = model.predict_classes(test_x, batch_size=128)
    # count = 0
    # t_y = []
    # for w2, r in zip(wyn2, test_y):
    #     # print(w1, w2, r)
    #     if r[0] == 1:
    #         t_y.append(0)
    #     elif r[1] == 1:
    #         t_y.append(1)
    #     elif r[2] == 1:
    #         t_y.append(2)
    #     elif r[3] == 1:
    #         t_y.append(3)
    #     if r[w2] == 1:
    #         count += 1
    # print(count / len(test_y) * 100, '%')
    #
    # conf = confusion_matrix(t_y, wyn2)
    # draw_confusion_matrix(conf, answer_key)
    # print(conf)

    with open("data/networks/networkFinal.pickle", 'rb') as file:
        model = pickle.load(file)
    model.summary()
    save_model(model, "data/networks/netFin.h5")
