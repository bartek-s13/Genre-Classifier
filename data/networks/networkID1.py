#  val_acc 70,77% 71,49% 70,54% (DICTIONARY 10000)
#  embedding 50, with stopwords

model = Sequential()
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH, trainable=True))
model.add(Conv1D(48, 3, activation='relu'))
model.add(AveragePooling1D(3))
model.add(Conv1D(46, 3, activation='relu'))
model.add(AveragePooling1D(3))
model.add(Conv1D(44, 3, activation='relu'))  # padding="same",
model.add(AveragePooling1D(20))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(len(answer_key), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
