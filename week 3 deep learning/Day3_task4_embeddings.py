import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load GloVe (about 65 MB, needs internet the first time)
model = api.load("glove-wiki-gigaword-50")

# 5 nearest words
print("Nearest to king:", model.most_similar("king", topn=5))
print("Nearest to computer:", model.most_similar("computer", topn=5))

# Analogy: king - man + woman
result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print("king - man + woman ->", result[0][0], "(top result)")

# Sentence vectors
def sv(sentence):
    vecs = [model[w] for w in sentence.lower().split() if w in model]
    return np.mean(vecs, axis=0)

s1 = "the cat sat on the mat"
s2 = "a kitten rested on the rug"
sim = cosine_similarity([sv(s1)], [sv(s2)])[0][0]
print("Sentence similarity:", round(sim, 3))

# Bonus: PCA plot of 30 words
words = ["king", "queen", "man", "woman", "prince", "princess",
         "paris", "france", "london", "england", "rome", "italy",
         "computer", "laptop", "software", "internet", "keyboard", "data",
         "dog", "cat", "horse", "lion", "tiger", "bird",
         "apple", "banana", "orange", "grape", "bread", "rice"]
pts = PCA(n_components=2).fit_transform([model[w] for w in words])
plt.figure(figsize=(10, 7))
plt.scatter(pts[:, 0], pts[:, 1])
for w, (x, y) in zip(words, pts):
    plt.annotate(w, (x, y))
plt.title("30 word vectors (PCA)")
plt.savefig("pca_words.png")
plt.show()

# Why embeddings capture meaning better than TF-IDF (4 lines):
# 1. Embeddings place similar words close together, so "car" and "automobile"
#    match, while TF-IDF treats them as completely unrelated words.
# 2. They are dense, low-dimensional vectors; TF-IDF is sparse and very large.
# 3. They learn meaning from context across huge amounts of text.
# 4. They support analogies and handle new phrasing better than word counts.