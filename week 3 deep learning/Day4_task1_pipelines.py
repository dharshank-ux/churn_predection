from transformers import AutoTokenizer, pipeline

# Tokenizer
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
for s in ["I love this app", "Payment failed twice", "Reset my password"]:
    enc = tok(s)
    print(tok.convert_ids_to_tokens(enc['input_ids']))
    print(enc['input_ids'])
    print(enc['attention_mask'])

# Sub-word split
print(tok.tokenize("unbelievability"))

# Sentiment
sents = ["Great service!", "Worst app ever", "It's okay", "Love the UI", "Payment keeps failing"]
for s, r in zip(sents, pipeline('sentiment-analysis')(sents)):
    print(s, r['label'], round(r['score'], 3))

# NER
ner = pipeline('ner', aggregation_strategy='simple')
for e in ner("Ravi from Yoho Technologies paid Rs. 45000 in Chennai on 12 March"):
    print(e['word'], e['entity_group'], round(e['score'], 3))

# Zero-shot
zs = pipeline('zero-shot-classification')
print(zs("I was charged twice this month", candidate_labels=["Billing", "Technical", "Account"]))