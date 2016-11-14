from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train,y_train)
print "Confusion matrix for Decision Tree Classifier:\n",confusion_matrix(y_test,clf1.predict(X_test))
m1 = confusion_matrix(y_test,clf1.predict(X_test))

clf2 = GaussianNB()
clf2.fit(X_train,y_train)
print "Confusion matrix for GaussianNB:\n",confusion_matrix(y_test,clf2.predict(X_test))
m2 = confusion_matrix(y_test,clf2.predict(X_test))


#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": m1,
 "Decision Tree": m2
}

## While the difference is fairly small for decision trees, naive Bayes seems to produce far more false negatives than false positives!
