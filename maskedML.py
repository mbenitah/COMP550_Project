import torch
from transformers.transformers import BertTokenizer, BertModel, BertForMaskedLM
from random import randrange

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)



# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input

text = '[CLS] I want to buy the car because it is cheap . I like to run everyday. Tables are round. [SEP]'
tokenized_text = tokenizer.tokenize(text)


#Determinig number of words to mask from percentage
percentage = 20
words_to_mask = int(percentage*0.01*len(tokenized_text))

# Mask a token that we will try to predict back with `BertForMaskedLM`
#masked_index = 10

#Masking chosen percentage of tokens from text
# masked = []
masked = {}
while words_to_mask>0:
    masked_index= randrange(len(tokenized_text))
    if masked_index not in masked:
        masked[masked_index] = tokenized_text[masked_index]
        tokenized_text[masked_index] = '[MASK]'
        # masked.append(masked_index)
        words_to_mask += -1

print(tokenized_text)
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

segments_ids = [0] * len(tokenized_text)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
# # ----------------------

# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased')

# # Set the model in evaluation mode to deactivate the DropOut modules
# # This is IMPORTANT to have reproducible results during evaluation!
# model.eval()

# # If you have a GPU, put everything on cuda
# use_cuda = False
# if use_cuda:
#     tokens_tensor = tokens_tensor.to('cuda')
#     segments_tensors = segments_tensors.to('cuda')
#     model.to('cuda')

# # Predict hidden states features for each layer
# with torch.no_grad():
#     # See the models docstrings for the detail of the inputs
#     outputs = model(tokens_tensor, token_type_ids=segments_tensors)
#     # Transformers models always output tuples.
#     # See the models docstrings for the detail of all the outputs
#     # In our case, the first element is the hidden state of the last layer of the Bert model
#     encoded_layers = outputs[0]
# # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
# assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
# # # -----------------------


# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
use_cuda = False
if use_cuda:
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

# Predict all tokens

#tokens_tensor = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
masks =  torch.tensor(-1 * torch.ones((tokens_tensor.shape[1]))).long()
masks[masked_index] = 0

print(masks)
 
with torch.no_grad():
    outputs = model(tokens_tensor, masked_lm_labels=masks)
    #outputs = model(tokens_tensor)
    predictions = outputs[1]
    loss, prediction_scores = outputs[:2]
#tokens_tensor = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1

print('loss: ', loss.item() )
print('prediction_scores: ', prediction_scores.shape)

words = {}

for index in masked:
    predicted_index = torch.argmax(predictions[0,index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # words.append(predicted_token)
    words[masked[index]] = predicted_token

print(words)

#print('predicted_token: ', predicted_token)