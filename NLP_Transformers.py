pip install transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

def chunk_text(text, max_length=512):
    tokens = tokenizer.encode(text, return_tensors='pt')[0]
    chunks = []

    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)

    return chunks

def generate_responses(chunks):
    responses = []
    for chunk in chunks:
        input_ids = chunk.unsqueeze(0)
        output = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=100)
        responses.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return responses

long_text = "Hello " * 50
long_text

chunks = chunk_text(long_text)
chunks

responses = generate_responses(chunks)
responses

for i, response in enumerate(responses):
    print(f"Response for chunk {i+1}:\n{response}\n")

