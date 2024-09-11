# GPT-2 Text Generation Example
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))

# BERT Embedding Extraction Example
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_text = "Transformers are great for NLP tasks"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
outputs = model(input_ids)

# Get hidden state embeddings
embeddings = outputs.last_hidden_state
print(embeddings)
