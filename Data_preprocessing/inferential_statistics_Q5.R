##question.1

####inferential statistics_Q5_Score

library(readxl)
library(readr)

###  import data set Assignment_module02(1)
plot1 <- as.data.frame(Assignment_module02_1_)

### select the column name as required 
### analysis on score column
p <- plot1$Points
hist(p)
a <- list("mean=", "median=","mode=")
b <- list("variance=", "std. deviation=", "minimum & maximum =", "range=")
c <- list("skewness=", "excess kurtosis=", "outliers=", "number of outliers=")

#custom mode function
mode <- function(m)
{
  m1 <- unique(m)
  m2 <- match(m,m1)
  m3 <- tabulate(m2)
  m4 <- max(m3)
  m5 <- m1[m4==m3]
}
md <- mode(p)
m1 <- unique(p)

#first moment business decision

a[[1]][2] <- mean(p)
a[[2]][2] <- median(p)
ml <- length(md)+1
for(i in 2:ml)
{
  a[[3]][i] <- md[i-1]
}

##no mode condition

nm <- length(m1)==length(md)
if(nm==TRUE)
{
  a[3] <- ("mode:-  all unique values are repeating equal number of times" )
}

#second moment business decision

b[[1]][2] <- var(p)
b[[2]][2] <- sd(p)
b[[3]][2:3] <- range(p,na.rm=FALSE)
maximum <- as.numeric(b[[3]][3])
minimum <- as.numeric(b[[3]][2])
b[[4]][2] <- maximum-minimum


#install e1071 package
library(e1071)

#third & fourth moment business decision


bx <- boxplot(p)

##skewness###
c1 <- skewness(p)
print(mean(p)) 
print(median(p))
s <- c(1:3) 
s[1]= mean(p)>median(p)
s[2]= mean(p)<median(p)
s[3]= mean(p)==median(p)
sk <- c("+ve skewness: because mean > median", "-ve skewness: because mean<median ", "normal or zero: mean = median ")
for(i in 1:3)
{
  if(s[i]==TRUE)
  {
    actual_skew <- sk[i]
  }
}

##kurtosis
c2 <- kurtosis(p)
k <- c(1:3)
k[1]= c2>0
k[2]= c2<0
k[3]= c2==0
ku <- c("leptokurtic", "platykurtic", "mesokurtic")
for(i in 1:3)
{
  if(k[i]==TRUE)
  {
    kurt <- ku[i]
  }
}

##outliers
c3 <- bx$out
c4 <- length(bx$out)+1
for(i in 2:c4)
{
  c[[3]][i] <- bx$out[i-1]
}
c[[1]][2]=c1
c[[2]][2]=c2
c[[4]][2]=c4-1


##normality checking

qq <- qqnorm(p)
qq1 <- qqline(p)


###conclusion

#first moment business decision
print("first moment business decisions are") 
print(a)

##second moment business decision
print("second moment business decisions are")
print(b)

##third & fourth moment business decisions & conclusions
print("third & fourth moment business decisions are")
print(c)
print("actual skewness is")
print(actual_skew)
print("type of kurtosis is")
print(kurt)


    