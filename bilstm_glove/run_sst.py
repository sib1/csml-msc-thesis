import numpy as np # linear algebra

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.initializers import Constant
import re
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def restrict_data_func(X_train, Y_train, num_each_label, num_labels):
  

  X_train_labelled = []
  Y_train_labelled = []

  for i in range(num_labels):
      X_train_labelled.append(X_train[Y_train == i][:num_each_label])
      Y_train_labelled.append(Y_train[Y_train == i][:num_each_label])
  X_train_labelled = np.concatenate(X_train_labelled, axis=0)
  Y_train_labelled = np.concatenate(Y_train_labelled, axis=0)

  return X_train_labelled, Y_train_labelled




train_labels = np.load('sst_train_labels.npy')
train_text = np.load('sst_train_text.npy')
valid_labels = np.load('sst_valid_labels.npy')
valid_text = np.load('sst_valid_text.npy')
test_labels = np.load('sst_test_labels.npy')
test_text = np.load('sst_test_text.npy')

#Shuffle Data to get a different sample each time

random_seed = 5
rand_state_1 = np.random.RandomState(random_seed)
shuffle_1 = rand_state_1.permutation(train_labels.shape[0])
train_labels = train_labels[shuffle_1]
train_text = train_text[shuffle_1]


#DECIDE HOW MANY OF EACH CLASS TO USE IN TRAINING DATA
restrict_training_data = 1 #Set equal to 0 or 1 
num_each_label=1000
num_labels = 5


if restrict_training_data:
  train_text, train_labels = restrict_data_func(train_text, train_labels, num_each_label,num_labels)


#print(test_text[700:710])
#print(test_labels[700:710])

#print(len(train_text))
#print(len(valid_text))
#print(len(test_text))

train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)


max_features = 20000 # this is the number of words we care about
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_text)

print('Tokenizing')

# this takes our sentences and replaces each word with an integer
X = tokenizer.texts_to_sequences(train_text)
X_val = tokenizer.texts_to_sequences(valid_text)
X_test = tokenizer.texts_to_sequences(test_text)


sequence_length = 80
# we then pad the sequences so they're all the same length (sequence_length)
X = pad_sequences(X, sequence_length)
y = train_labels
X_test = pad_sequences(X_test, sequence_length)
y_test = test_labels
X_val = pad_sequences(X_val, sequence_length)
y_val = valid_labels
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 300

"""

UNCOMMENT THIS SECTION TO USE GLOVE AND CHANGE THE EMBEDDING LAYER IN MODEL TO THE ONE WHICH USES THE GLOVE MATRIX

print('Loading Glove')

embeddings_index = {}
f = open('glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))



print('Constructing Glove Matrix')

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))


#print(embeddings_index.get(word))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():

    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)


"""

model = Sequential()
#model.add(Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=sequence_length,trainable=True))
model.add(Embedding(num_words, 128, input_length=sequence_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation='softmax'))


model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
batch_size = 64
history = model.fit(X, y, epochs=3, batch_size=batch_size, verbose=1, validation_data=[X_val, y_val])
print("Evaluating")

res = model.evaluate(X_test,y_test)
print(res[1])




