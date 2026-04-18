# Class 1: Sports & Athletics (Context: Winning/Medals)
doc1 = "The gold medal price is high effort"
doc2 = "Winning a gold medal needs a high jump"
doc3 = "Market for a gold medal is a trade of sweat"
doc4 = "The athlete will trade all for a gold medal"

# Class 2: Finance & Economy (Context: Market/Investment)
doc5 = "The gold bars price is high today"
doc6 = "Investing in gold bars needs a high rate"
doc7 = "Market for gold bars is a trade of money"
doc8 = "The bank will trade all for gold bars"

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix  

def preprocess_text(text):
    """
    Make sure to lowercase and remove punctuation.
    """
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    tokens = text.split()
    return tokens


def vectorize(docs, n_gram_size=1):
    cleaned_docs = [' '.join(preprocess_text(doc)) for doc in docs]
    vectorizer = CountVectorizer(ngram_range=(n_gram_size, n_gram_size))
    X = vectorizer.fit_transform(cleaned_docs)
    return X


def evaluate_clusters(labels, true_labels):
    labels = np.array(labels)
    true_labels = np.array(true_labels)
    inverted = 1 - labels
    if np.mean(inverted == true_labels) > np.mean(labels == true_labels):
        labels = inverted

    accuracy = np.mean(labels == true_labels)
    predicted_positive = labels == 1
    true_positive = (labels == 1) & (true_labels == 1)
    precision = np.sum(true_positive) / np.sum(predicted_positive) if np.sum(predicted_positive) else 0.0
    return accuracy, precision, labels

# Training / Clustering

all_docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]

# 1-gram Experiment
X1 = vectorize(all_docs, n_gram_size=1)
km1 = KMeans(n_clusters=2, random_state=42).fit(X1)

# 2-gram Experiment
X2 = vectorize(all_docs, n_gram_size=2)
km2 = KMeans(n_clusters=2, random_state=42).fit(X2)

print(f"1-gram clusters: {km1.labels_}")
print(f"2-gram clusters: {km2.labels_}")

true_labels = [0, 0, 0, 0, 1, 1, 1, 1]
acc1, prec1, aligned1 = evaluate_clusters(km1.labels_, true_labels)
acc2, prec2, aligned2 = evaluate_clusters(km2.labels_, true_labels)

cm1 = confusion_matrix(true_labels, aligned1)
cm2 = confusion_matrix(true_labels, aligned2)

print("\n1-gram evaluation:")
print(f"Accuracy: {acc1:.2f}, Precision: {prec1:.2f}")
print("Confusion Matrix:")
print(cm1)

print("\n2-gram evaluation:")
print(f"Accuracy: {acc2:.2f}, Precision: {prec2:.2f}")
print("Confusion Matrix:")
print(cm2)
