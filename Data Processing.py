
# coding: utf-8

# In[1]:

import csv
import nltk
from tqdm import tqdm
import numpy as np
from keras.preprocessing import sequence


# In[2]:

# How many lines to process
line_limit = 400000


# In[3]:
with open('../train_new.csv', 'r') as f:
    reader = csv.reader(f)
    
    lines = []
    
    for row in tqdm(reader):
        lines += [row]
        
    lines = lines[1:line_limit+1]


# In[4]:

X = []
Y = []
unicode_errors = 0

for label, sentence in tqdm(lines):
    try:
        Y += [label]
        X += [nltk.word_tokenize(sentence.decode('ascii', 'ignore'))]
    except UnicodeDecodeError:
        unicode_errors += 1
        print 'Unicode errors:', unicode_errors


# In[5]:

counts = dict()

for i in tqdm(range(len(X))):
    for word in X[i]:
        word = word.lower()
        if word not in counts:
            counts[word] = 0
        counts[word] += 1


# In[6]:

freq_sorted = [k for v, k in sorted([(v, k) for k, v in counts.iteritems()], reverse=True)]


# In[8]:

vocabulary = ['out_of_vocab'] + freq_sorted
print len(vocabulary)


# In[9]:

idx2w = [(i, v) for i, v in zip(range(len(vocabulary)), vocabulary)]
w2idx = [(v, i) for i, v in idx2w]

w2idx = dict(w2idx)
idx2w = dict(idx2w)


# In[10]:

X_vec = []
Y_vec = []

for i in tqdm(range(len(X))):
    X_vec += [[]]
    Y_vec += [np.zeros(3)]
    
    for w in X[i]:
        if w.lower() not in vocabulary:
            X_vec[-1] += [w2idx['out_of_vocab']]
        else:
            X_vec[-1] += [w2idx[w.lower()]]
    
    Y_vec[-1][int(Y[i])] = 1


# In[11]:

X_vec = sequence.pad_sequences(X_vec, maxlen=300)
Y_vec = np.array(Y_vec)


# In[12]:

print 'Vocabulary size:', len(vocabulary)


# In[13]:

np.save("X.npy", X_vec)
np.save("Y.npy", Y_vec)


# ========================================================================

# In[14]:

embeddings_index = {}
f = open('/home/soham/DL-NLP/Assignment2/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[15]:

vocab_size = len(vocabulary)
embedding_dim = 300

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in w2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        if i >= vocab_size:
            continue
        
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[16]:

np.save("embedding_matrix.npy", embedding_matrix)

# ========================================================================

# In[17]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
import numpy as np


# In[18]:

vocab_size = 72669
maxlen = 300


# In[19]:

embedding_matrix = np.load("embedding_matrix.npy")


# In[23]:

model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=maxlen, weights=[embedding_matrix], trainable=True))
model.add(Bidirectional(GRU(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(Bidirectional(GRU(100, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


# In[24]:

mcp = ModelCheckpoint('best.hdf5', monitor="val_acc", save_best_only=True, save_weights_only=False)


# In[ ]:

x_train = np.load("X.npy")
y_train = np.load("Y.npy")

model.fit(x_train, y_train, batch_size=128, epochs=20, callbacks=[mcp], validation_split=0.05)

