
from nltk.stem import PorterStemmer
import string
# get the text file and stopwords
from os import listdir
import numpy as np
stopwords = []
path = 'stopwords.txt'
with open(path) as f:
    stopwords = f.read().splitlines()

# remove line change 
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
        if char not in string.punctuation:
            data += char


    # #tokenize the words
    words = data.split(" ")
    result = []
    # stemming and removing stopwords in the end
    for a in words:  
        a = PorterStemmer().stem(a)
        if(a != ""):
            result.append(a)
    return result

FILE_PATH = "./data/"

files = listdir(FILE_PATH)
files = [f for f in files if f[0] != "."]
files.sort(key=lambda x: int(x[:-4]))
doc = []
for text in files:
    with open(FILE_PATH + text, "r") as f:
        d_id = str(text)[:-4]
        document = f.read()
        doc.append([d_id, document])

termFrequecies = []
docFrequecies = {}
for a in doc:
    id, document = a
    words = tokenize(document)
    term ={}
    for word in words:
        if word in term:
            term[word] +=1
        else:
            term[word] =1
    set_words = set(words)
    for i in set_words:
        if i in docFrequecies:
            docFrequecies[i] +=1
        else:
            docFrequecies[i] =1
    termFrequecies.append([id, term])
docFrequecies = dict(sorted(docFrequecies.items(), key=lambda x: x[0]))

index = {}
idx = 0
for term in docFrequecies:
    index[term] = idx
    idx +=1

# with open("./output/dictionary.txt", "w") as f:
#     f.write("t_index\t term\tdf\n")  
#     for key in docFrequecies:
#         idx = index[key]
#         term = key
#         df = docFrequecies[key]
#         f.write(f"{idx}  {term}  {df}\n")

tf_idf_vector = {} 

for a in termFrequecies:
    times = 0
    id, term_list = a
    tf_idf_list = []
    for i in term_list:
        tf_idf = term_list[i] * np.log(len(files)/docFrequecies[i])
        tf_idf_list.append([index[i], tf_idf])
    tf_idf_list = np.array(tf_idf_list)
    tf_idf_list[:,1] = tf_idf_list[:,1]/np.linalg.norm(tf_idf_list[:,1])
    tf_idf_list = tf_idf_list[tf_idf_list[:, 0].argsort()]
    tf_idf_vector[id] = tf_idf_list

array = []
for a in termFrequecies:
    id, term_list = a
    # with open(f"./output/{id}.txt", "w") as f:
    #     f.write(f"{len(tf_idf_vector[id])}\n")
    #     f.write("t_index\t tf-idf\n")  # Head
    #     for a in tf_idf_vector[id]:
    #         f.write(f"{int(a[0])}  {a[1]}\n")
    temp = np.zeros(len(index))
    for a in tf_idf_vector[id]:
        temp[int(a[0])-1] = a[1]
    array.append([temp])


def cosine(id_x, id_y):
    return np.matmul(np.array(array[id_x-1]), np.array(array[id_y-1]).T).item()
    
print(cosine(1,2))

I = np.ones(len(files))
C = np.zeros((len(files), len(files)))
for i in range(len(files)):
    for j in range(i+1, len(files)):
        C[i][j] = cosine(i+1, j+1)
        C[j][i] = C[i][j]
print(C[0][1])
print(C[1][0])

def max_pair(C, I, size):
    max = -1
    doc_i, doc_m = -1, -1
    for i in range(size):
        if I[i] != 1:
            continue
        for m in range(i+1, size):
            if I[m] == 1 and i != m:
                if max < C[i][m]:
                    max = C[i][m]
                    doc_i, doc_m = i, m
    return doc_i, doc_m

A = []
for k in range(len(files)-1):
    i, m = max_pair(C, I, len(files))
    A.append([i ,m])
    for j in range(len(files)):
        C[i][j] = min(cosine(i, j), cosine(m, j))
        C[j][i] = min(cosine(j, i), cosine(j, m))
    I[m] = 0
diction = {str(i) : [i] for i in range(len(files))}
for doc_i, doc_m in A:
    new_element = diction[str(doc_m)]
    diction.pop(str(doc_m))
    diction[str(doc_i)] += new_element
    if len(diction) == 20:
        with open(f"./20.txt", "w") as f:
            for k, v in diction.items():
                for id in sorted(v):
                    f.write(f"{id+1}\n")
                f.write("\n")
    if len(diction) == 13:
        with open(f"./13.txt", "w") as f:
            for k, v in diction.items():
                for id in sorted(v):
                    f.write(f"{id+1}\n")
                f.write("\n")
    if len(diction) == 8:
        with open(f"./8.txt", "w") as f:
            for k, v in diction.items():
                for id in sorted(v):
                    f.write(f"{id+1}\n")
                f.write("\n")