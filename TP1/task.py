#Zerhouni Ghizlene Houria
#4INGIA
#hope that the fakenews dataset is in the same directory as the task.py file, if not, please change the path in the code to the correct one
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import pandas as pd
import string

nltk.download('stopwords')
#text cleaning
#dans ce cas, on va supprimer que le point, la virgule et les deux poins d'exclamation et d'interrogation
D1 = "I feel good."     
D2 = "I do feel even better now, thank you!"
D3 = "What's happened to you? you all good?"

def textCleaning(text):
    text = text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    return text 
d1=textCleaning(D1)
d2=textCleaning(D2)
d3=textCleaning(D3)

def textTokenization(text):
    tokens = text.split()
    return tokens


#Text normalization
#pour la normalisation, je vais mettre tout en miniscule, faire la tokenisation et le text cleaning,
#Stopword Removal,
def textNormalization(text):
    normalized_text = text.lower()
    normalized_text = textTokenization(normalized_text)
    stopwordsremover = set(stopwords.words('english'))
    normalized_text = [word for word in normalized_text if word not in stopwordsremover]
    return normalized_text
d111=textNormalization(d1)
d222=textNormalization(d2)
d333=textNormalization(d3)
print('Text Normalization:')
print(d111)
print(d222)
print(d333)

D1 = ['feel', 'good']
D2 = ['feel', 'better', 'thank']
D3 = ['happened', 'good']
#Vectorizing
def vectorize(D1,D2,D3):
 unique_words = set(D1 + D2 + D3)
 print('Unique words:', unique_words)
 d1_vector = []
 d2_vector = []
 d3_vector = []

 for word in unique_words:
    if word in D1:
        d1_vector.append(1)
    else:
        d1_vector.append(0)
 print('D1 vector:', d1_vector)
 for word in unique_words:
    if word in D2:
        d2_vector.append(1)
    else:
        d2_vector.append(0)
 print('D2 vector:', d2_vector)
 for word in unique_words:       
    if word in D3:
        d3_vector.append(1)
    else:
        d3_vector.append(0)
 print('D3 vector:', d3_vector)
print(vectorize(D1,D2,D3))
#naive bayes

poswords=['feel', 'good']
negwords=['feel', 'bad']
doc = [
    "feel good",   # D1
    "feel bad"     # D2
]
labels = [
    "positive",
    "negative"
]


def naiveBayes(text):
    """
    Naive Bayes utilisant la loi de Bayes:
    P(classe|texte) = P(texte|classe) * P(classe) / P(texte)
    """
    words = text.lower().split()
    
    p_positive = 1 / 2  
    p_negative = 1 / 2  
    
    p_words_positive = 1
    for word in words:
        if word in poswords:
            p_words_positive *= 0.8  
        else:
            p_words_positive *= 0.2  
    
    
    p_words_negative = 1
    for word in words:
        if word in negwords:
            p_words_negative *= 0.8  
            p_words_negative *= 0.2  
    
    # P(texte|classe) = P(mots|classe)
    p_text_given_positive = p_words_positive * p_positive
    p_text_given_negative = p_words_negative * p_negative
    
  
    if p_text_given_positive > p_text_given_negative:
        return "positive"
    else:
        return "negative"

print("Prediction for 'feel good':", naiveBayes("feel good"))
print("Prediction for 'feel bad':", naiveBayes("feel bad"))

#Evaluation task
print("\nEvaluation:")
#This part was made by IA pour pouvoir lire les fichiers csv de la database
base_dir = os.path.dirname(os.path.abspath(__file__))
true_path = os.path.join(base_dir, "fakeNews", "News _dataset", "True.csv")
fake_path = os.path.join(base_dir, "fakeNews", "News _dataset", "Fake.csv")
missing = [p for p in (true_path, fake_path) if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing file(s): {', '.join(missing)}")


true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)

true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

dataset = pd.concat([true_df, fake_df])
dataset = dataset.sample(frac=1)

def clean_text(text):
    cleaned = textCleaning(text)
    normalized = textNormalization(cleaned)
    return normalized
#print("text cleaned and normalized:")
dataset["text"] = dataset["text"].apply(clean_text)
#print('Vectorization:')
#texts_list = dataset["text"].tolist()
#vectorize(*texts_list[:3])

print("\nNaive Bayes Predictions:")
for i in range(min(5, len(dataset))):  
    text = dataset["text"].iloc[i]
    true_label = dataset["label"].iloc[i]
    predicted_label = naiveBayes(" ".join(text)) 
    print(f"Texte {i+1}: {true_label} -> Prédit: {predicted_label}")


