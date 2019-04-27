import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from sklearn import tree

traindata = pd.read_csv('train.csv', header=0)
  
def numNon(column, trainset):
    trainset[column].fillna(trainset[column].median(skipna=True), inplace=True)

def strNon (column, trainset):
    c = Counter(trainset[column])
    str=list(c.elements())
    s=str[0]
    trainset[column].fillna(trainset[column].value_counts().idxmax(), inplace=True)

def factorizze (column,dataset):
    labels, uniques = pd.factorize(dataset[column], sort=True)
    dataset[column] = labels
    return labels

def get_acc(columns):
    x_train=traindata[columns]
    y_train = traindata["Survived"]
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=.3, random_state=2)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy = %2.3f" % accuracy_score(Y_test, y_pred))

def testsomedata(columns , testdata):
    X_train = traindata[columns]
    Y_train = traindata["Survived"]
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(testdata[columns])
    print(y_pred)
    row_index = 0
    with open("output.csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(("PassengerId", "Survived"))
        for entries in y_pred:
            writer.writerow((testdata["PassengerId"][row_index], str(entries)))
            row_index += 1


traindata.drop(traindata.columns[[0,3,8,10]], axis=1, inplace=True)
strNon("Embarked",traindata)
numNon("Age",traindata)
strNon("Sex",traindata)
strNon("SibSp",traindata)
strNon("Pclass",traindata)
strNon("Parch",traindata)
numNon("Fare",traindata)
factorizze("Embarked", traindata)
factorizze("Sex",traindata)
get_acc(["Embarked","Age","Sex","SibSp","Pclass","Parch","Fare"])
get_acc(["Embarked","Age","Sex","Pclass","Parch"])
get_acc(["Age","Sex","SibSp","Pclass","Parch","Fare"])
get_acc(["Embarked","Age","Sex","SibSp","Pclass","Fare"])
get_acc(["Embarked","Sex","SibSp","Pclass","Parch","Fare"])
get_acc(["Age","Sex"])

testdata = pd.read_csv('test.csv', header=0)
testdata.drop(testdata.columns[[2,7,9]], axis=1, inplace=True)
strNon("Embarked",testdata)
strNon("Pclass",testdata)
strNon("Sex",testdata)
strNon("SibSp" , testdata)
strNon("Parch",testdata)
numNon("Fare",testdata)
numNon("Age",testdata)
factorizze("Embarked", testdata)
factorizze("Sex",testdata)
testsomedata(["Embarked","Age","Sex","SibSp","Pclass","Parch","Fare"],testdata)