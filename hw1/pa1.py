import requests
from nltk.stem import PorterStemmer
# get the text file and stopwords
content = requests.get("https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt")
stopwords = []
path = 'stopwords.txt'
with open(path) as f:
    stopwords = f.read().splitlines()

# remove line change 
c1 = content.text.replace("\r\n","")
doc = c1.split(" ")
c1 = ""
#first check stopwords and change letters to little
for a in doc:
    a = a.lower()
    if a not in stopwords:
        c1 += (a+" ")

# # remove punctuations 
punc =['!','"', '#' ,'$' ,'%' ,'&' ,"'", '(', ')' ,'*' ,'+' , '-',',', '.', '/', ':', ';', '?', '@', '[' , ']', '^', '_', '`', '{', '|' ,'}', '~',']']
data = ""
for char in c1:
    if char not in punc:
        data += char


# #tokenize the words
words = data.split(" ")

result = []

# stemming and removing stopwords in the end
for a in words:  
    a = PorterStemmer().stem(a)
    result.append(a)

# Save the result as a txt file
with open("result.txt", "w") as file:
    for term in result:
        file.write(term + "\n")

