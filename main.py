# load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import (
        train_test_split,
        cross_val_score,
        StratifiedKFold
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load the dataset 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# high level veiw of data
dimensions = dataset.shape
summary = dataset.describe()
group_by_class = dataset.groupby('class').size()

print(dimensions)
print(summary)
print(group_by_class)

# extend veiw of data with visualizations
# looking at 2 different plots -->
 
# Univariate plot to better understand each attribute
# Multivariate plot to better understand the relationships between attributes

# TODO: research Univariate and Multivariate plots

# Start with some univariate plots (plots of each individual variable)
# box and whisker plots --> TODO: what are these and how are they read
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# We can also create a histogram of each input variable 
# to get a better idea of distribution
# TODO: Gaussian distribution --> what is that? 
dataset.hist()
pyplot.show()

# Interactions between variables
# Scatter plots help to spot structured relationships 
# between variables
scatter_matrix(dataset)
pyplot.show()

 # Split-out validation dataset
array = dataset.values
X = array [:,0:4] # how did we decide on slicing values? 
y = array [:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Test harness section 
# Stratified 10-fold cross validation to estimate model accuracy
# This wil split our dataset into 10 parts, train on 9 and test on 1

# NOTE: Stratified means the each fold or split of the dataset will 
# aim to have the same distribution of exampl by class as exist in the 
# whole training set. 

# NOTE: % accuracy == (num of correctly predicted instances / total instances) * 100

# Build Model

# Testing 6 algorithms and determine which has the highest accuracy

# LINEAR
# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)

# NON LINEAR
# K-Nearest Neighbors (KNN)
# Classification and Regression Trees (CART)
# Gaussian Naive Bayes (GNB)
# Support Vector Machines (SVM)

# Spot Check algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
# running it a few times

results = []
names = []  
for name, model in models: 
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# A good way to compare results is create a box and whisker plot for each distribution 
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm comparison')
pyplot.show()

# Make predictions! 
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))