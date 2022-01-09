##### imdb kaidhi film review #####

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
oneplus_reviews=[]

## all the reviews selected from a single page
ip=[]  
url="https://www.imdb.com/title/tt9900782/reviews?ref_=tt_urv"
response = requests.get(url)
soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
reviews = soup.find_all("div",attrs={"class":"text show-more__control"})      
for i in range(len(reviews)):
    ip.append(reviews[i].text) 
    
oneplus_reviews=oneplus_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews
                                    # but here all the reviews are from same page: even though step is added but doesnt go for the loop 
# writng reviews in a text file 
with open("oneplus.txt","w",encoding='utf8') as output:
    output.write(str(oneplus_reviews))
	
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(oneplus_reviews)
ip_rev_string

import nltk
from nltk.corpus import stopwords

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)
ip_rev_string

# words that contained in iphone XR reviews
ip_reviews_words = ip_rev_string.split(" ")
ip_reviews_words

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)

with open("G://txt mng & nlp//Datasets NLP//stopwords_en.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend(["karthi","lokesh","paruthiveeran","kanagaraj","film","actor","movie"])

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph without stop words
ip_rev_string = " ".join(ip_reviews_words)
ip_rev_string

# WordCloud can be performed on the string inputs.
# Corpus level word cloud

wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("G://txt mng & nlp//Datasets NLP//positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open("G://txt mng & nlp//Datasets NLP//negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

# wordcloud with bigram
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ["karthi","movie","role","cinema","role","edge","seat","lokesh","paruthiveeran","kanagaraj","police station","actor","movie","director","daughter","film","make","life","tamil","injured","cop","doesnt","doesn", "meet","police","released","attempt","shooter",'month','thought','bought','months','night','back','days','thought','time','much','used','doesnt','review','feel','full','looks','takes','new','oneplus','will','even','plus','goes','like','still','got','dont','get','one','use','ised','phonesi','yrs','cant','just','also','using','since','overall','say','can','really','now',] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()