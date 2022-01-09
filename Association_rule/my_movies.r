######### my movies #########

# Load the dataset
library(readr)
input <- read_csv(file.choose())
mydata1 <- input

# eliminate non_numeric & less informative datas
mydata <- mydata1[,c(6:15)]

## DATA CLEANING AND EDA BEGINS

# missing data checking
sum(is.na(mydata))    ## no null values

## outlier treatment
# since data is binary standardized form wouldn't gonna have no outliers 

# install arules pakages for building association rules
# install.packages("arules")
library("arules") # Used for building association rules i.e. apriori algorithm

# summary
head(mydata)
summary(mydata)

# converting to matrix format for rules formation
qq <- as.matrix(mydata)

# the matrix data set is converting to transaction form by eliminating '0' showing no purchasing of perticular product
qq <- as(qq,"transactions")

# making rules using apriori algorithm 
# Keep changing support and confidence values to obtain different rules

# Building rules using apriori algorithm
arules <- apriori(qq, parameter = list(support = 0.11, confidence = .6, minlen = 2))
arules  # 12 rules

# Viewing rules based on lift value
inspect(head(sort(arules, by = "lift"))) # to view we use inspect 

# Overal quality 
head(quality(arules))

# install.packages("arueslViz")
library("arulesViz") # for visualizing rules

# Different Ways of Visualizing Rules
plot(arules)

windows()
plot(arules, method = "grouped")
plot(arules[1:10], method = "graph") # for good visualization try plotting only few rules

write(arules, file = "a_rules.csv", sep = ",")

getwd()

