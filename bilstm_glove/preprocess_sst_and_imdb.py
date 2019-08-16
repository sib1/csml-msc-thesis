from torchnlp.datasets import imdb_dataset, smt_dataset
import numpy as np 

def create_IMDB_labels(data,num_train):
  labels_tensor = np.zeros(num_train)
  labels = data.__getitem__('sentiment')
  labels = labels[0:num_train]

  pos_indices = [i for i, x in enumerate(labels) if x == "pos"]
  neg_indices = [i for i, x in enumerate(labels) if x == "neg"]

  labels_tensor[pos_indices] = int(1)
  labels_tensor[neg_indices] = int(0)
  
  return labels_tensor

def create_SMT_labels(data,num_train):
  labels_tensor = np.zeros(num_train)
  labels = data.__getitem__('label')
  labels = labels[0:num_train]
  
  very_pos_indices = [i for i, x in enumerate(labels) if x == "very positive"]
  pos_indices = [i for i, x in enumerate(labels) if x == "positive"]
  neut_indices = [i for i, x in enumerate(labels) if x == "neutral"]
  neg_indices = [i for i, x in enumerate(labels) if x == "negative"]
  very_neg_indices = [i for i, x in enumerate(labels) if x == "very negative"]

  labels_tensor[very_pos_indices] = 0
  labels_tensor[pos_indices] = 1
  labels_tensor[neut_indices] = 2
  labels_tensor[neg_indices] = 3
  labels_tensor[very_neg_indices] = 4
  
  return labels_tensor



train = smt_dataset(train=True, fine_grained=True)
valid = smt_dataset(dev=True, fine_grained=True)
test = smt_dataset(test=True, fine_grained=True)


train_labels = create_SMT_labels(train,len(train))
train_text = np.array(train.__getitem__('text'))
valid_labels = create_SMT_labels(valid,len(valid))
valid_text = np.array(valid.__getitem__('text'))
test_labels = create_SMT_labels(test,len(test))
test_text = np.array(test.__getitem__('text'))

np.save('sst_train_text',train_text)
np.save('sst_train_labels',train_labels)
np.save('sst_valid_text',valid_text)
np.save('sst_valid_labels',valid_labels)
np.save('sst_test_text',test_text)
np.save('sst_test_labels',test_labels)

train = imdb_dataset(train=True)
test = imdb_dataset(test=True)

train_labels = create_IMDB_labels(train,len(train))
test_labels = create_IMDB_labels(test,len(test))
train_text = np.array(train.__getitem__('text'))
test_text = np.array(test.__getitem__('text'))

np.save('imdb_train_text',train_text)
np.save('imdb_train_labels',train_labels)
np.save('imdb_test_text',test_text)
np.save('imdb_test_labels',test_labels)