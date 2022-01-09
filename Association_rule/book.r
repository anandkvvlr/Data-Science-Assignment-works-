######### book #########

# Load the dataset
library(readr)
input <- read_csv(file.choose())
mydata <- input

## 3.1) DATA CLEANING & PREPROCESSING 

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

# 4.1) APPLICATION OF APRIORI ALGORITHM
arules <- apriori(qq, parameter = list(support = 0.08, confidence = 0.75, minlen = 2))
arules  ## 39 rules

# 4.2) BUILD MOST FREQUENT ITEM SET AND PLOT THE RULE
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

