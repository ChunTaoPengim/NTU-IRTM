from nltk.stem import PorterStemmer
import string
# get the text file and stopwords
from os import listdir
import numpy as np
import pandas as pd
import math
stopwords = []
#i want to use the stopwords.txt in the same folder
# how to txt file in same folder
# but I got FileNotfounderror
# so I use the absolute path
path = "hw3/stopwords.txt"
with open(path) as f:
    stopwords = f.read().splitlines()


def tokenize(text):
    c1 = text.replace("\r\n","")
    c1 = c1.replace("\n","")
    doc = c1.split(" ")
    c1 = ""
    #first check stopwords and change letters to little
    for a in doc:
        a = a.lower()
        if a not in stopwords:
            c1 += (a+" ")

    # # remove punctuations 

    data = ""
    for char in c1:
        if char not in string.punctuation and not char.isdigit():
            data += char


    # #tokenize the words
    words = data.split(" ")
    result = []
    # stemming and removing stopwords in the end
    for a in words :  
        if a not in stopwords:
            a = PorterStemmer().stem(a)
            if(a != "" and a not in stopwords and len(a) > 1):
                result.append(a)
    return result


FILE_PATH = "hw3/data/files/"
files = listdir(FILE_PATH)
files = [f for f in files if f[0] != "."]
files.sort(key=lambda x: int(x[:-4]))
doc = []
for text in files:
    with open(FILE_PATH + text, "r") as f:
        d_id = str(text)[:-4]
        document = f.read()
        doc.append([d_id, document])
labels = {}

with open("hw3/training.txt", "r") as f:
    for line in f:
        a =[]
        a = line.split(" ")
        labels[a[0]] = a[1:-1]

labels["13"].append("541")
print(labels["13"])

training_set = list()
for le in labels:
    for id in labels[le]:
        training_set.append(doc[int(id)-1]+[le])

columns = ['doc_id', 'document', 'label']
df_train = pd.DataFrame(columns=columns)
i = 0
for le in labels:
    for id in labels[le]:
        df_train.loc[i] = [int(id), doc[int(id) - 1][1], le]
        i += 1

df_train = df_train.astype({"doc_id":int,"label":int})
df_train = df_train.sort_values(by="doc_id")
df_train = df_train.reset_index(drop=True)


df_test = pd.DataFrame(columns=columns)
i = 0
for d in doc:
    doc_id = int(d[0])
    if doc_id not in df_train["doc_id"].values:
        df_test.loc[i] = [doc_id, doc[int(doc_id) - 1][1], None]
        i += 1
df_test = df_test.astype({"doc_id":int})
df_test = df_test.sort_values(by="doc_id")
df_test = df_test.reset_index(drop=True)
print(df_train.shape)
print(df_test.shape)

def get_tf(corpus: list):
    # Term frequency list
    # A list which store the TF dictionary of each document
    tf_list = list()
    
    # Iterate the corpus
    for document in corpus:
        
        # Words of each document
        document_word_list = tokenize(document)
        
        # Unique words and its frequency of each document
        tf = dict()
        for word in document_word_list:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        # Add this dictionary into the global list
        tf_list.append(tf)
    
    return tf_list
df_train["tf"] = get_tf(df_train["document"])  
df_train = df_train[["doc_id", "document", "tf", "label"]]  
df_test["tf"] = get_tf(df_test["document"])  
df_test = df_test[["doc_id", "document", "tf", "label"]]  
def find_unique_tokens(token_lists) -> list:
    unique_tokens = set()
    for token_list in token_lists:
        for token in token_list:
            unique_tokens.add(token)
    return list(unique_tokens)

def likelihood(C: dict, D: pd.DataFrame):
    vocabulary = find_unique_tokens(D.tf)
    N = len(D)
    chi2 = dict()
    count = 0
    for term in vocabulary:
        chi2_term = 0
        matrix = dict()
        matrix["tp"] = D[D["tf"].apply(lambda x: term in x)]
        matrix["ta"] = D[D["tf"].apply(lambda x: term not in x)]
        chi2_class = []
        for c in C:
            matrix["cp"] = D[D["label"] == int(c)]
            matrix["ca"] = D[D["label"] != int(c)]
            n11 = len(matrix["tp"][matrix["tp"]["label"] == int(c)])
            n01 = len(matrix["tp"][matrix["tp"]["label"] != int(c)])
            n10 = len(matrix["ta"][matrix["ta"]["label"] == int(c)])
            n00 = len(matrix["ta"][matrix["ta"]["label"] != int(c)])
            total = n11 + n01 + n10 + n00
            
            r = float((n11 + n01) / total)
            r2 = float(n11 / (n11 + n10))
            r3 = float(n01 / (n00 + n01))
            ratio1 = (r**n11) * (r**n01) * ((1-r)**n10) * ((1-r)**n00)
            ratio2 = (r2**n11) * ((1-r2)**n10)  * (r3**n01)* ((1-r3)**n00)
            scores = -2*math.log(ratio1 / ratio2)  
            chi2_class.append(scores)
            chi2_term = max(chi2_class)
        chi2[term] = chi2_term
        count += 1
        if count % 500 == 0:
            print(f"training at : {count}")
    vocabulary = sorted(chi2, key=chi2.get, reverse=True)[:450] 
    return vocabulary
vocabulary = likelihood(labels, df_train)
print(f"feature selection size: {len(vocabulary)}")
def train_multinominal_nb(C: dict, D: pd.DataFrame, vocabulary):
    n_docs = len(D)
    prior = dict()
    cond_prob = {term: dict() for term in vocabulary}
    
    for c in C:
        n_class_docs = len(C[c])
        class_docs = D[D["label"] == int(c)]
        tct = dict()
        # 事前機率
        prior[c] = n_class_docs / n_docs
        # 事後機率
        for term in vocabulary:
            tokens_of_term = 0
            for tf in class_docs["tf"]:
                if term in tf:
                    tokens_of_term += tf[term]
            tct[term] = tokens_of_term
        for term in vocabulary:
            cond_prob[term][c] = (tct[term]+1) / (sum(tct.values())+len(vocabulary))
            
    return vocabulary, prior, cond_prob

def test_multinomial_nb(document, C, vocabulary, prior, cond_prob):
    tf = document["tf"]
    score = dict()
    
    for c in C:
        score[c] = math.log2(prior[c])
        for term in tf:
            if term in vocabulary:
                score[c] += (math.log2(cond_prob[term][c]))*(tf[term])
                # Bug: 因為取 log 是相加，因此出現多次時不應該用次方算，應該用相乘才對
            
    return max(score, key=score.get)
vocabulary, prior, cond_prob = train_multinominal_nb(labels, df_train, vocabulary)
df_test["label"] = df_test.apply(
    test_multinomial_nb, C=labels, vocabulary=vocabulary, prior=prior, cond_prob=cond_prob, axis=1)
print(df_test.shape)

df_test = df_test.sort_values(by="doc_id")
df_test = df_test.reset_index(drop=True)
output_df = df_test[["doc_id", "label"]]
output_df.columns = ["Id", "Value"]
output_df.to_csv("hw3/output/output.csv", index=False)