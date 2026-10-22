from sklearn.datasets import load_breast_cancer
df = load_breast_cancer(as_frame=True).frame
df.to_csv('data.csv', index=False)
print("Saved", df.shape)