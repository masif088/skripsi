lemmatized_text=['lihat', 'ada', 'sebut', 'kami', 'dari', 'imaji', 'sociopreneur', 'sama', 'yayasan', 'mimpi', 'indonesia', 'gagas', 'buah', 'gera', 'yang', 'kami', 'beri', 'nama', 'tanam', 'buku']

stopword_file = open("stopword.txt", "r")
#Source = https://www.ranks.nl/stopwords

lots_of_stopwords = []

for line in stopword_file.readlines():
    lots_of_stopwords.append(str(line.strip()))

stopwords_plus = []
stopwords_plus = lots_of_stopwords
stopwords_plus = set(stopwords_plus)
#print(stopwords_plus)
processed_text = []
for word in lemmatized_text:
    if word not in stopwords_plus:
        processed_text.append(word)
print (processed_text)
