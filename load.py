from transformers.transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers.transformers import BertTokenizer, BertModel, BertForMaskedLM

output_dir = "./output/"

# Re-load the saved model and vocabulary

# Example for a Bert model
model = BertForMaskedLM.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)  # Add specific options if needed
