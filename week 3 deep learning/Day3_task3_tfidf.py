import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

tickets = [
    "I was charged twice for my subscription",
    "The app crashes when I open settings",
    "Please reset my password, cannot log in",
    "Refund not received for cancelled order",
    "Error 500 when uploading a file",
    "How do I change my registered email",
]

# Preprocess: lowercase, remove punctuation, tokenise
clean = [re.sub(r"[^\w\s]", "", t.lower()) for t in tickets]
tokens = [t.split() for t in clean]
print("Tokens:", tokens[0])

# TF-IDF
vec = TfidfVectorizer()
X = vec.fit_transform(clean)
print("Shape:", X.shape)
row = X[0].toarray().ravel()
words = vec.get_feature_names_out()
print("Top 5 words ticket 1:", list(words[row.argsort()[::-1][:5]]))

# Cosine similarity
q = vec.transform(["cannot log in to my account"])
sims = cosine_similarity(q, X).ravel()
print("Most similar:", tickets[sims.argmax()], "| score:", round(sims.max(), 3))

# Classifier on 30 tickets
data = [
    ("I was charged twice for my subscription", "billing"),
    ("Refund not received for cancelled order", "billing"),
    ("Please refund my last payment", "billing"),
    ("My invoice shows the wrong amount", "billing"),
    ("Credit card was billed after cancellation", "billing"),
    ("Why was I charged an extra fee", "billing"),
    ("I need a receipt for my payment", "billing"),
    ("Discount code was not applied at checkout", "billing"),
    ("Update my billing address for invoices", "billing"),
    ("Payment failed but money was deducted", "billing"),
    ("The app crashes when I open settings", "bug"),
    ("Error 500 when uploading a file", "bug"),
    ("Screen freezes on the home page", "bug"),
    ("Button does not respond when I click it", "bug"),
    ("App closes suddenly after update", "bug"),
    ("Page shows a blank screen", "bug"),
    ("Upload fails with an error message", "bug"),
    ("Notifications are not working", "bug"),
    ("App is very slow and keeps lagging", "bug"),
    ("Search feature shows an error", "bug"),
    ("Please reset my password, cannot log in", "account"),
    ("How do I change my registered email", "account"),
    ("I forgot my username", "account"),
    ("Cannot sign in to my account", "account"),
    ("Delete my account permanently", "account"),
    ("Update my phone number on profile", "account"),
    ("Account locked after failed login", "account"),
    ("Verification code is not arriving", "account"),
    ("How do I change my display name", "account"),
    ("Enable two factor login on my account", "account"),
]
texts = [d[0] for d in data]
labels = [d[1] for d in data]

Xtr, Xte, ytr, yte = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42)
v = TfidfVectorizer()
clf = LogisticRegression(max_iter=1000)
clf.fit(v.fit_transform(Xtr), ytr)
print("Accuracy:", clf.score(v.transform(Xte), yte))

# TF-IDF explanation (3 lines):
# 1. TF-IDF measures how important a word is to one document in a collection.
# 2. TF counts how often the word appears in that document; IDF lowers the
#    weight of words that appear in many documents.
# 3. So distinctive words score high and common filler words score low.