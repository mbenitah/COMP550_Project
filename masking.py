import torch
from transformers.transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from random import randrange
import numpy as np
import os

CONFIG_FILE = 'config.json'
SAVE_DIRECTORY = './output/'

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def mask(text, percentage):
    '''Masks words in a given text.
    Inputs:
    text:       text with words to be masked
    percentage: percentage of words that will be masked
                in text
    output:
    tokenized_text:     list of tokens of the original text
                        with some words replaced to [MASK]
    masked:             dictionary of indexes and words that have been replaced
                        with [MASK]
    ids:                ids of encoded text'''

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input
    # text = '[CLS] I want to buy the car because it is cheap . I like to run everyday. Tables are round. [SEP]'
    # tokenized_text = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(ids)

    # Determinig number of words to mask from percentage
    words_to_mask = int(percentage * len(tokenized_text))

    # Masking chosen percentage of tokens from text
    # masked = []
    masked = {}
    while words_to_mask > 0:
        masked_index = randrange(len(tokenized_text))
        if masked_index not in masked:
            if tokenized_text[masked_index] not in tokenizer.all_special_tokens:
                masked[masked_index] = tokenized_text[masked_index]
                tokenized_text[masked_index] = '[MASK]'
                # masked.append(masked_index)
                words_to_mask += -1

    # assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

    print(tokenized_text)
    print(masked)
    return tokenized_text, masked, ids


def predict(tokenized_text, masked=None):
    '''Predicts masked words in a tokenized text.
    Input:
    tokenized_text:     list of tokens of the original text
                        with some words replaced to [MASK]
    masked:             dictionary of indexes and words that have been replaced
                        with [MASK] (for calculating accuracy)
    '''

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    ids_tensor = torch.tensor([ids])
    # segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # model.config_class(os.path.join(SAVE_DIRECTORY, CONFIG_FILE))
    # model.config_class(max_position_embeddings=1024)
    model.eval()


    # If you have a GPU, put everything on cuda
    if torch.cuda.is_available():
        ids_tensor = ids_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        model.to('cuda')

    # Creates a vector of -1 equal to text length
    # and marks the position of each masked word with 0
    masks = torch.tensor(-1 * torch.ones((ids_tensor.shape[1]))).long()
    masks[list(masked.keys())] = 0
    # print(masks)

    with torch.no_grad():
        outputs = model(ids_tensor, masked_lm_labels=masks)
        loss, predictions = outputs[:2]

    print('loss: ', loss.item())
    print('prediction_scores: ', predictions.shape)

    words = {}
    scores = {}

    acc = 0
    for index in masked:
        predicted_index = torch.argmax(predictions[0, index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        # words.append(predicted_token)
        words[masked[index]] = predicted_token
        scores[masked[index]] = max(predictions[0, index])
        if masked[index] == predicted_token:
            acc += 1
    acc = acc / len(masked)
    print('Accuracy: {:.2%}'.format(acc))
    print('Average score: {:.2f}'.format(np.average(list(scores.values()))))

    print(words)
    print(scores)

    #print('predicted_token: ', predicted_token)

    
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


def main():
    tokenized_text, masked, ids = mask("I went to the opera yesterday. It was a great experience. The singer was crashed by the chandelier. She sang a high note.", 0.2)
    predict(tokenized_text, masked)

if __name__ == '__main__':
    main()