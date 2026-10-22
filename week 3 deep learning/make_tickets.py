import random
import pandas as pd

random.seed(42)

templates = {
    "billing": [
        ("i was charged twice for order {n}", "high"),
        ("my card was declined when paying for order {n}", "medium"),
        ("please send the invoice for order {n}", "low"),
    ],
    "technical": [
        ("the app crashes every time i open it, ticket {n}", "high"),
        ("i cannot log in after the latest update, ref {n}", "high"),
        ("the dark mode setting does not save, ref {n}", "low"),
    ],
    "shipping": [
        ("my package for order {n} has not arrived yet", "medium"),
        ("order {n} was delivered to the wrong address", "high"),
        ("can i change the delivery date for order {n}", "low"),
    ],
    "account": [
        ("i forgot my password and need a reset, ref {n}", "medium"),
        ("someone accessed my account without permission, ref {n}", "high"),
        ("how do i change my email address, ref {n}", "low"),
    ],
    "refund": [
        ("i want a refund for order {n}, item was damaged", "high"),
        ("how long does a refund take for order {n}", "low"),
        ("refund for order {n} still not received", "medium"),
    ],
}
prefixes = ["", "hi, ", "hello team, ", "urgent: ", "please help. ", "hey, "]
suffixes = ["", " thanks", " please help", " this is frustrating", " any update?"]

rows = []
for _ in range(700):
    cat = random.choice(list(templates))
    text, pri = random.choice(templates[cat])
    text = random.choice(prefixes) + text.format(n=random.randint(1000, 99999)) + random.choice(suffixes)
    rows.append((text.upper() if random.random() < 0.1 else text, cat, pri))

pd.DataFrame(rows, columns=["text", "category", "priority"]).to_csv("tickets.csv", index=False)
print("saved tickets.csv")