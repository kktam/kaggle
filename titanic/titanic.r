# Kaggle Titanic
# based on blog post "How to perform a Logistic Regression in R"
# https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/

df <- read.csv('train.csv',header=T,na.strings=c(""))
# print(df)

na.blank <- function(x) {
  x[is.na(x)] <- " "
  return (x)
}

na.noclass <- function(x) {
  x[is.na(x)] <- 0
  return (x)
}

na.patch <- function(x) {
   sum(is.na(x))
}

Mode <- function(x) {
  ux <- unique(x);
  tab <- tabulate(match(x, ux)); 
  ux[tab == max(tab)];
}

sapply(df, na.patch)

sapply(df, function(x) length(unique(x)))

library(Amelia)
missmap(df, main = "Missing values vs observed")

#allcolumns <- c('PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked')
#data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10,12))
data <- subset(df,select=c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'))

#print(data)

data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)

is.factor(data$Sex)

is.factor(data$Embarked)

# get mode
embarkedMode <- Mode(data$Embarked)

# patch missing data with column mode
data <- data[!is.na(data$Embarked),]
data[is.na(data$Embarked)] <- embarkedMode
#rownames(data) <- NULL


# check data again after patching, to make sure there are no more missing values
df2 <- as.data.frame.matrix(data)
df2<-`names<-`(df2, c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'))
missmap(df2, main = "Missing values vs observed after fixing")

usekfoldvalidation <- 0
k <- 81

if (usekfoldvalidation == 1) {

	#Randomly shuffle the data
	data<-data[sample(nrow(data)),]

	#Create 10 equally size folds
	folds <- cut(seq(1,nrow(data)),breaks=k,labels=FALSE)

	#Perform 10 fold cross validation
	for(i in 1:k){
		#Segement your data by fold using the which() function 
		testIndexes <- which(folds==i,arr.ind=TRUE)
		train <- data[testIndexes, ]
		test <- data[-testIndexes, ]
	}
} else {

	train <- data[1:800,]
	test <- data[801:889,]
	
}

model <- glm(Survived ~.,family=binomial(link='logit'),data=train)

summary(model)

anova(model, test="Chisq")

#install.packages("pscl")
library(pscl)
pR2(model)

fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))

# ROC curve
library(ROCR)
p <- predict(model, newdata=subset(test,select=c(2,3,4,5,6,7,8)), type="response")
pr <- prediction(p, test$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, col=rainbow(10))

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
