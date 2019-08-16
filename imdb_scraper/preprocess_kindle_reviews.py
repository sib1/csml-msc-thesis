import numpy as np
import torch.nn as nn
import glob
import json
import torch
import math
from pytorch_transformers import BertForMaskedLM, BertTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from random import Random, randrange, shuffle


def padding(sequences, maxlen=None, dtype=np.int, padding='post', value=0.):
     
    # padding = "pre" pads from the front, "post" pads at the end
    # Function pads to convert lists of strings (of uneven length) into numpy arrays
  
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    padded = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if padding == 'post':
            padded[idx, :len(s)] = s
        elif padding == 'pre':
            padded[idx, -len(s):] = s
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return padded
  
#Function to pre-process data for BERT
def process_bert(data,max_len):
  
  #Import Bert tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
  #Create arrays to store our tokenised training data
  tokens_ids = []
  input_masks = []
  masked_labels = []
  
  num_train = len(data)
  #Tokenise the training data, add CLS/SEP tokens and then convert to IDs
  for i in range(num_train):

  	#Tokenize Data
    text = data[i]
    token = tokenizer.tokenize(text)
    token_len = len(token)

    if(i%1000==0):
    	print(i)
    	print(str(i*100/num_train) + '%' + ' complete')

    #Split into smaller chunks if longer than max_len
    
    for i in range(math.ceil(token_len/max_len)):
	    if token_len > max_len:
		    partial = token[i*max_len:(i+1)*max_len]
	    else:
		    partial = token

	    partial_len = len(partial)
	    
	    #Randomy choose 15% of tokens
	    mask_idx = np.random.choice(partial_len, math.ceil(partial_len*0.15))
	    token_numpy = np.array(partial)

	    #Get the labels for the words to be masked
	    masked_words = token_numpy[mask_idx]
	    masked_words_ids = tokenizer.convert_tokens_to_ids(masked_words)

	    #Convert 80% of masked tokens to [MASK], leave 20% the same
	    num_to_mask = math.ceil(len(mask_idx)*0.8)
	    mask_idx_replace = mask_idx[0:num_to_mask]
	    
	    #Repalce 80% of chosen tokens with [MASK]
	    token_numpy[mask_idx_replace] = '[MASK]'
	    partial = token_numpy.tolist()

	    #Insert Bert Specific start and end tokens
	    partial.insert(0,'[CLS]')
	    partial.append('[SEP]')

	    #Create Inputs masks
	    input_mask = [1] * len(partial)

	    #Convert tokens to IDs
	    token_id = tokenizer.convert_tokens_to_ids(partial)

	    #Create masked labels, -1 everywhere except the masked tokens
	    masked_label = [-1] * (len(partial) - 2)
	    masked_label_numpy = np.array(masked_label)
	    masked_label_numpy[mask_idx] = masked_words_ids
	    masked_label = masked_label_numpy.tolist()
	    masked_label.insert(0,-1)
	    masked_label.append(-1)

	    input_masks.append(input_mask)
	    tokens_ids.append(token_id)
	    masked_labels.append(masked_label)
    

  #Pad the training data so everything is of uniform length
  tokens_ids = padding(tokens_ids)
  input_masks = padding(input_masks)
  masked_labels = padding(masked_labels)

  max_length = len(tokens_ids[0])
  num_train = len(tokens_ids)

  #Convert training data and labels to tensors
  ids_tensor = torch.tensor([tokens_ids], dtype=torch.long).resize_((num_train,max_length))
  input_masks_tensor = torch.tensor([input_masks], dtype=torch.long).resize_((num_train,max_length))
  masked_labels_tensor = torch.tensor([masked_labels], dtype=torch.long).resize_((num_train,max_length))
  return ids_tensor, input_masks_tensor, masked_labels_tensor




reviews = []

with open('reviews_Kindle_Store_5.json') as f:
    for line in f:
        reviews.append(json.loads(line)['reviewText'])






ids_tensor, input_masks_tensor, masked_labels_tensor = process_bert(reviews[0:300000],126)

torch.save(ids_tensor, 'amazon_random_ids_128')
torch.save(input_masks_tensor, 'amazon_random_input_masks_128')
torch.save(masked_labels_tensor, 'amazon_random_masked_labels_128')




