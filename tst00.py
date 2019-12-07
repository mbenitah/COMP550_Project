import torch
from transformers.transformers import BertTokenizer, BertModel, BertForMaskedLM
from random import randrange
import json
import os

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'

SAVE_DIRECTORY = './output/'
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input

text = 'I want to buy the car because it is cheap . I like to run everyday. Tables are round.'

# if SPECIAL_TOKENS_MAP_FILE is not None:
#     special_tokens_map_file = os.path.join(SAVE_DIRECTORY, SPECIAL_TOKENS_MAP_FILE)
#     special_tokens_map = json.load(open(special_tokens_map_file, encoding="utf-8"))
#     tokenizer.add_special_tokens(special_tokens_map)

ids = tokenizer.encode(text, add_special_tokens=True)
tokenized_text = tokenizer.convert_ids_to_tokens(ids)
#tokenized_text = tokenizer.tokenize(text, add_special_tokens=True)
print(tokenized_text)