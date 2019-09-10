#groceries data set
groceries<-read.transactions(file.choose(),format="basket")
inspect(groceries[1:10])
class(groceries)

library(arules)
data("Groceries")
summary(Groceries)
inspect(Groceries[1:10])
rules <- apriori(Groceries,parameter = list(support = 0.002,confidence = 0.05,minlen=5))
inspect(rules[1:5])
windows()
plot(rules,method = "scatterplot")
plot(rules,method = "grouped")
plot(rules,method = "graph")


rules <- sort(rules,by="lift")

#phonedata data set

phonedata<-read.transactions(file.choose(),format="basket")
inspect(phonedata[1:10])
class(phonedata)

library(arules)
summary(phonedata)
inspect(phonedata[1:10])
rules <- apriori(phonedata,parameter = list(support=0.002,confidence=0.5,minlen=2))
inspect(rules[1:5])
windows()
plot(rules,method = "scatterplot")
plot(rules,method = "grouped")
plot(rules,method = "graph")


rules <- sort(rules,by="lift")

inspect(rules[1:4])
