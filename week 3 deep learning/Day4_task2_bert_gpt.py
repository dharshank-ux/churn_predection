from transformers import pipeline

# Part 1: BERT fill-mask
fill = pipeline('fill-mask', model='bert-base-uncased')
for s in ["The capital of France is [MASK].",
          "I forgot my [MASK] so I cannot log in.",
          "The doctor gave me some [MASK] for the fever."]:
    best = fill(s)[0]
    print(s, "->", best['token_str'], round(best['score'], 3))

# Part 2: GPT-2 text generation
gen = pipeline('text-generation', model='gpt2')
for p in ["Dear customer, your refund",
          "To reset your password, you should",
          "The best way to learn AI is"]:
    out = gen(p, do_sample=True, temperature=0.7, max_new_tokens=30)
    print(out[0]['generated_text'])
    print("---")

# Part 3: Count parameters
for name in ['bert-base-uncased', 'gpt2']:
    m = pipeline('feature-extraction', model=name).model
    print(name, sum(p.numel() for p in m.parameters()))
    
    # Comparison table: BERT vs GPT
# | Feature       | BERT (bert-base-uncased)         | GPT (gpt2)                        |
# |---------------|----------------------------------|-----------------------------------|
# | Architecture  | Encoder only                     | Decoder only                      |
# | Reads/writes  | Reads whole sentence both ways   | Writes left to right, word by word|
# | Best for      | Understanding text               | Generating text                   |
# | Example tasks | Classification, NER, fill-mask   | Replies, stories, autocomplete    |
# | Parameters    | 109,482,240 (about 109M)         | 124,439,808 (about 124M)          |

# Why BERT for tickets and GPT for replies (4 lines):
# 1. BERT reads the full ticket in both directions, so it understands context well.
# 2. Classification needs understanding a label, not writing text, which suits an encoder.
# 3. GPT is trained to predict the next word, so it writes fluent, natural replies.
# 4. So: BERT sorts the ticket, GPT drafts the response.