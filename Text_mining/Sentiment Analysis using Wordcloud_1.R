
###################################################################
######### Amazon Reviews Extraction ###########
#install.packages("rvest")
#install.packages("XML")
#install.packages("magrittr")
#install.packages("wordcloud")
#install.packages("ROAuth")
#install.packages("base64enc")
#install.packages("httpuv")
#install.packages("stringr")
# build a corpus using the text mining (tm) package
#install.packages
#library(stringr)
#library(base64enc)
#library(ROAuth)
#library(httpuv)
library(tm)
library(wordcloud)
library(rvest)
library(XML)
library(magrittr)

######### Amazon URL ###########
aurl <- "https://www.amazon.in/OnePlus-Midnight-Black-128GB-Storage/product-reviews/B07DJHY82F/ref=cm_cr_getr_d_paging_btm_prev_1?showViewpoints=1&pageNumber="

amazon_reviews <- NULL

for (i in 1:30){
     murl <- read_html(as.character(paste(aurl,i,sep="")))
     rev <- murl %>% html_nodes(".review-text") %>% html_text()
     amazon_reviews <- c(amazon_reviews,rev)
}

?html_nodes
?html_text
?read_html
write.table(amazon_reviews,"Oneplus6T.txt")
getwd()

############################
#### Sentiment Analysis ####
txt <- amazon_reviews

str(txt)
length(txt)
View(txt)

#install.packages("tm")
library(tm)

# Convert the character data to corpus type
x <- Corpus(VectorSource(txt))

inspect(x[1])

x <- tm_map(x, function(x) iconv(enc2utf8(x), sub='byte'))
?tm_map

# Data Cleansing
x1 <- tm_map(x, tolower) 
inspect(x1[1])

x1 <- tm_map(x1, removePunctuation)
inspect(x1[1])

inspect(x1[5])
x1 <- tm_map(x1, removeNumbers)
inspect(x1[1])

x1 <- tm_map(x1, removeWords, stopwords('english'))
inspect(x1[1])

# striping white spaces 
x1 <- tm_map(x1, stripWhitespace)
inspect(x1[1])

# Term document matrix 
# converting unstructured data to structured format using TDM

tdm <- TermDocumentMatrix(x1)
dtm <- t(tdm) # transpose   ## creatm dtm from tdm
dtm <- DocumentTermMatrix(x1)  ## directly creating tdm

# To remove sparse entries upon a specific value
corpus.dtm.frequent <- removeSparseTerms(tdm, 0.99) 
?removeSparseTerms

tdm <- as.matrix(tdm)
dim(tdm)

tdm[1:20, 1:20]

inspect(x[1])

# Bar plot
w <- rowSums(tdm)
w

w_sub <- subset(w, w >= 20)
w_sub

barplot(w_sub, las=2, col = rainbow(30))

# Term repeats maximum number of times
x1 <- tm_map(x1, removeWords, c('phone','got','one','day','print','phones','product','amazon','buy','android','unlock','compared','month','thought','bought','months','night','back','days','thought','time','much','used','doesnt','review','feel','full','looks','takes','new','oneplus','will','even','plus','goes','like','still','got','dont','get','one','use','ised','phonesi','yrs','cant','just','also','using','since','overall','say','can','really','now'))
x1 <- tm_map(x1, stripWhitespace)

tdm <- TermDocumentMatrix(x1)

tdm <- as.matrix(tdm)
tdm[100:109, 1:20]

# Bar plot after removal of the term 'phone','oneplus','will','goes','like',,still','got','ised','phonesi','yrs','cant','just','also','using','since','overall','say'
w <- rowSums(tdm)
w

w_sub <- subset(w, w >= 45)
w_sub

barplot(w_sub, las=2, col = rainbow(30))

##### Word cloud #####
#install.packages("wordcloud")
library(wordcloud)

wordcloud(words = names(w_sub), freq = w_sub)

w_sub1 <- sort(rowSums(tdm), decreasing = TRUE)
head(w_sub1)

wordcloud(words = names(w_sub1), freq = w_sub1) # all words are considered

# better visualization
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F, colors=rainbow(30), scale = c(2,0.5), rot.per = 0.4)
windows()
wordcloud(words = names(w_sub), freq = w_sub, random.order=F, colors= rainbow(30),scale=c(3,0.5),rot.per=0.3)
windows()
?wordcloud

############# Wordcloud2 ###############
### for different visualization effect

#installed.packages("wordcloud2")
library(wordcloud2)

w1 <- data.frame(names(w_sub), w_sub)
colnames(w1) <- c('word', 'freq')

wordcloud2(w1, size=0.3, shape='circle')
?wordcloud2

wordcloud2(w1, size=0.3, shape = 'triangle')
wordcloud2(w1, size=0.3, shape = 'star')


#### Bigram ####
#install RWeka & rjava
##### since having package issues couldnt do bigram workcloud ###
#library(RWeka) # for bigram option
#library(rjava) # for bigram option
#library(wordcloud)

#minfreq_bigram <- 2
#bitoken <- NGramTokenizer(x1, Weka_control(min = 2, max = 2))
#two_word <- data.frame(table(bitoken))
#sort_two <- two_word[order(two_word$Freq, decreasing = TRUE), ]

#wordcloud(sort_two$bitoken, sort_two$Freq, random.order = F, scale = c(2, 0.35), min.freq = minfreq_bigram, colors = brewer.pal(8, "Dark2"), max.words = 150)

#####################################
# lOADING Positive and Negative words  
pos.words <- readLines(file.choose())	# read-in positive-words.txt
neg.words <- readLines(file.choose()) 	# read-in negative-words.txt

stopwdrds <-  readLines(file.choose())

### Positive word cloud ###
pos.matches <- match(names(w_sub1), pos.words)
pos.matches <- !is.na(pos.matches)
freq_pos <- w_sub1[pos.matches]
names <- names(freq_pos)
windows()
wordcloud(names, freq_pos, scale=c(4,1), colors = brewer.pal(8,"Dark2"))


### Matching Negative words ###
neg.matches <- match(names(w_sub1), neg.words)
neg.matches <- !is.na(neg.matches)
freq_neg <- w_sub1[neg.matches]
names <- names(freq_neg)
windows()
wordcloud(names, freq_neg, scale=c(4,.5), colors = brewer.pal(8, "Dark2"))
