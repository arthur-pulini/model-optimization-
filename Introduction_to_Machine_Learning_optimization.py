import pandas as pd
import numpy as np
import graphviz
import seaborn as sns
from datetime import datetime
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

SEED = 301
np.random.seed(SEED)

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
datas = pd.read_csv(uri)
print (datas)

change = {
    'no' : 0,
    'yes' : 1
}
datas.sold = datas.sold.map(change)

currentYear = datetime.today().year
datas['model_age'] = currentYear - datas.model_year
head = datas.head()
print(head)

datas['Kilometers_per_year'] = datas.mileage_per_year * 1.0934
head = datas.head()
print(head)

datas = datas.drop(columns = ['Unnamed: 0', 'mileage_per_year', 'model_year'], axis = 1)
head = datas.head()
print(head)

x = datas[['price', 'model_age', 'Kilometers_per_year']]
y = datas['sold']

cv = 10
model = DummyClassifier()
results = cross_validate(model, x, y, cv = cv)
average = results['test_score'].mean()
standardDeviation = results['test_score'].std()
print("Range accuracy DummyClassifier = [%.2f, %.2f]" % ((average - 2 * standardDeviation) * 100 , (average + 2 * standardDeviation) * 100))

model = DecisionTreeClassifier(max_depth = 2)
results = cross_validate(model, x, y, cv = cv)
average = results['test_score'].mean()
standardDeviation = results['test_score'].std()
print("Range accuracy DecisionTreeClassifier = [%.2f, %.2f]" % ((average - 2 * standardDeviation) * 100 , (average + 2 * standardDeviation) * 100))

#Simulando modelos de carros aleatórios para poder usar o GroupKfold
np.random.seed(SEED)
datas['model'] = datas.model_age + np.random.randint(-2, 3, size=10000)
print(datas.model.unique())

def printResults (results):
    average = results['test_score'].mean()
    standardDeviation = results['test_score'].std()
    print("Test results = ", results['test_score'])
    print("Average accuracy = %.2f" % (average * 100))
    print("Range accuracy = [%.2f, %.2f]" % ((average - 2 * standardDeviation) * 100 , (average + 2 * standardDeviation) * 100))

#Usando o cross_validate no DecisionTreeClassifier
cv = GroupKFold(n_splits=10)
decisionTreeModel = DecisionTreeClassifier(max_depth=2)
results = cross_validate(decisionTreeModel, x, y, cv = cv, groups = datas.model)
printResults (results)

#Usando o cross_validate no SVC
scaler = StandardScaler()
model = SVC()

pipeline = Pipeline([('transformation', scaler), ('estimator', model)])

cv = GroupKFold(n_splits=10)
results = cross_validate(model, x, y, cv = cv, groups = datas.model)
printResults (results)

#Imprimindo a decision tree
decisionTreeModel.fit(x, y)

features = x.columns
dotData = export_graphviz(decisionTreeModel, out_file=None, filled=True, rounded=True, class_names=['no', 'yes'], feature_names=features)
graph = graphviz.Source(dotData)
#graph.view()

#Definindo parâmetros
def DecisionTree (max_depth):
    cv = GroupKFold(n_splits=10)
    decisionTreeModel = DecisionTreeClassifier(max_depth=max_depth)
    results = cross_validate(decisionTreeModel, x, y, cv = cv, groups = datas.model, return_train_score=True)
    trainScore = results['train_score'].mean() * 100
    testScore = results['test_score'].mean() * 100
    print("max_depth: %d, average train_score: %.2f, average test_score: %.2f" % (max_depth, results['train_score'].mean() * 100, results['test_score'].mean() * 100))
    table = [max_depth, trainScore, testScore]
    return table

#Analisando gracicamente o desempenho do test relacionado ao treino
print("Testing diferents max_depth: \n\n")
averagesResult = [DecisionTree(i) for i in range(1,33)]
averagesResult = pd.DataFrame(averagesResult, columns = ['max_depth', 'train', 'test'])
print(averagesResult.head())

sns.lineplot(x = 'max_depth', y= 'train', data= averagesResult)
sns.lineplot(x = 'max_depth', y= 'test', data= averagesResult)
plt.legend(["train", "test"])
#plt.show()

print(averagesResult.sort_values("test", ascending=False))
