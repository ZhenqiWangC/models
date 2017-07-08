from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2
import re
from nltk.corpus import stopwords
import operator
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

class Node:
    def __init__(self,var_index,split_value,left,right):
        self.var_index = var_index
        self.split_value = split_value
        self.left = left
        self.right = right


def gini_index(nodes,class_values):
    gini = 0.0
    for value in class_values:
        for node in nodes:
            size =len(node)
            if size == 0:
                continue
            prob = [row[-1] for row in node].count(value)/float(size)
            gini += (prob) * (1-prob) * float(size)
    return gini


def single_split(var_index,value,dataset):
    left = list()
    right = list()
    for row in dataset:
        if row[var_index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def one_zero_loss(true,pred):
    return sum([1 for i in zip(true,pred) if i[0] == i[1]]) / float(len(true))


def best_split(dataset,random_feature=False,variable_size=0.5):
    class_values = list(set([row[-1] for row in dataset]))
    best_index, split_value, best_score, left, right = 0, 0, 1e10, None, None
    if random_feature:
        var_indeces= np.random.choice(int(len(dataset[0]) - 1), int(variable_size*(len(dataset[0]) - 1)))
    else:
        var_indeces =range(len(dataset[0]) - 1)
    for var_index in var_indeces:
        split_values = list(set([row[var_index] for row in dataset]))
        for value in split_values:
            groups = single_split(var_index, value, dataset)
            impurity = gini_index(groups, class_values)
            if impurity < best_score:
                best_index, split_value, best_score, left, right = var_index, value, impurity, groups[0], groups[1]
    return best_index, split_value, impurity, left, right


def recursive_split(dataset, max_depth, min_size, depth,random_feature=False,variable_size=0.5):
    var_index, split_value, impurity, left, right = best_split(dataset, random_feature,variable_size )
    if impurity == 0 or max_depth < depth or len (dataset) < min_size:
        if len(left) == 0:
            left = right
        if len(right) == 0:
            right = left
        left = max(set([row[-1] for row in left]), key=left.count)
        right = max(set([row[-1] for row in right]), key=right.count)
        return Node(var_index, split_value, left, right)
    else:
        if len(left) == 0:
            left = right
        if len(right) == 0:
            right = left
        return Node(var_index, split_value, \
                    recursive_split(left, max_depth, min_size, depth+1, random_feature,variable_size),\
                    recursive_split(right, max_depth, min_size, depth+1, random_feature,variable_size ))


def decision_tree(dataset, max_depth, min_size,random_feature=False,variable_size=0.5):
    var_index, split_value, impurity, left, right = best_split(dataset,random_feature,variable_size)
    if len(left) == 0:
        left = right;
    if len(right) == 0:
        right = left
    return Node(var_index, split_value, recursive_split(left, max_depth, min_size, 1, random_feature,variable_size),\
                recursive_split(right, max_depth, min_size, 1, random_feature,variable_size))


def random_forest(dataset, max_depth, min_size, num_tree, sample_size, variable_size, var_names):
    forest = dict()
    dataset = np.array(dataset)
    roots = dict()
    for i in range(num_tree):
        sam_in = np.random.choice(len(dataset),sample_size, replace=False)
        temp_data = dataset[sam_in,:].tolist()
        forest[i] = decision_tree(temp_data,max_depth,min_size,random_feature=True,variable_size=variable_size)
        if var_names[forest[i].var_index]+'<'+str(forest[i].split_value) in roots:
            roots[var_names[forest[i].var_index]+'<'+str(forest[i].split_value)] += 1
        else:
            roots[var_names[forest[i].var_index]+'<'+str(forest[i].split_value)] = 1
    return forest, sorted(roots.items(), key=lambda x: x[1],reverse=True)


def single_predict(node, row):
    if row[node.var_index] < node.split_value:
        if isinstance(node.left, Node):
            return single_predict(node.left, row)
        else:
            return node.left
    else:
        if isinstance(node.right, Node):
            return single_predict(node.right, row)
        else:
            return node.right


def predict(tree, dataset):
    pred = list()
    for row in dataset:
        pre = single_predict(tree,row)
        pred.append(pre)
    return pred


def single_predict_path(node, row, var_names):
    if row[node.var_index] < node.split_value:
        print var_names[node.var_index], "<", node.split_value
        if isinstance(node.left, Node):
            return single_predict_path(node.left, row, var_names)
        else:
            print "Classified to Class", node.left
            return node.left
    else:
        print var_names[node.var_index], ">=", node.split_value
        if isinstance(node.right, Node):
            return single_predict_path(node.right, row, var_names)
        else:
            print "Classified to Class", node.right
            return node.right


def randf_predict(forests,dataset):
    predictions = list()
    for i in range(len(forests)):
        pred = predict(forests[i],dataset)
        predictions.append(pred)
    return stats.mode(np.array(predictions))[0][0]


def print_tree(tree,vars,depth):
    print "depth:", depth, "Split variable",vars[tree.var_index],"Split value:", tree.split_value
    if isinstance(tree.left,Node):
        print "left"
        print_tree(tree.left,vars,depth+1)
    else:
        print tree.left
    if isinstance(tree.right,Node):
        print "right"
        print_tree(tree.right,vars,depth+1)
    else:
        print tree.right




def load_txt(file):
    file = open(file, 'r')
    text = file.read().lower()
    file.close()
    text = text.replace('\r\n', ' ')
    text= re.sub("[^a-z$!:?</@#%([\*]", " ", text)
    words = text.split()
    stop = set(stopwords.words('english'))
    words = [i for i in words if i not in stop]
    return " ".join(words)


def txt_to_array(file_match):
    text_array = []
    onlyfiles =glob.glob(file_match)
    for i in range(len(onlyfiles)):
        text_array.append(load_txt(onlyfiles[i]))
    return text_array


def test_to_array():
    text_array = []
    for i in range(10000):
        text_array.append(load_txt("dist/test/"+str(i)+".txt"))
    return text_array

def load_census():
    training = pd.read_csv('./hw5_census_dist/train_data.csv',na_values=["?"])
    test = pd.read_csv('./hw5_census_dist/test_data.csv',na_values=["?"])
    var_names = list(training)[:-1]
    y = training['label']
    x = training.ix[:,:-1]
    for col in var_names:
        if np.issubdtype(test[col].dtype, np.number):
            test[col].fillna(test[col].mean())
            x[col].fillna(x[col].mean())
        else:
            test[col].fillna(test[col].mode()[0])
            x[col].fillna(x[col].mode()[0])
    x = pd.get_dummies(x)
    test = pd.get_dummies(test)
    common_var = list(set(x).intersection(test))
    x = x[common_var]
    test = test[common_var]
    return x, y,test,common_var


def load_spam(k=2000):
    ham_x = txt_to_array("dist/ham/*.txt")
    spam_x = txt_to_array("dist/spam/*.txt")
    test_x = test_to_array()
    label = np.zeros(len(ham_x))
    label = np.concatenate((label, np.ones(len(spam_x))))
    x = np.concatenate((ham_x, spam_x))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", lowercase=False)
    vectorizer = vectorizer.fit(x.tolist())
    train_data_features = vectorizer.transform(x.tolist())
    fselect = SelectKBest(chi2, k=k)
    train_data_features = fselect.fit_transform(train_data_features, label)
    var_names = np.asarray(vectorizer.get_feature_names())[fselect.get_support()]
    train_data_features = train_data_features.toarray()
    test = vectorizer.transform(test_x)
    test = fselect.transform(test)
    test = test.toarray()
    return train_data_features,label,test,var_names

def load_titanic():
    train = pd.read_csv('./hw5_titanic_dist/titanic_training.csv', na_values="")
    test = pd.read_csv('./hw5_titanic_dist/titanic_testing_data.csv', na_values="")
    train.drop(train.index[[705]], inplace=True)
    train['t_class'] = ""
    train['t_no'] = ""
    for i in range(len(train['ticket'])):
        words = str(train['ticket'].iloc[i]).split(" ")
        if words[-1].isdigit():
            train['t_no'].iloc[i] = words[-1]
        else:
            train['t_no'].iloc[i] = 0
        if len(words) == 1:
            train['t_class'].iloc[i] = 'no'
        else:
            train['t_class'].iloc[i] = re.sub("[^a-zA-Z0-9]", "", "".join(words[:-1]))
    train['t_no'] = train['t_no'].astype(float)
    train['t_class'] = train['t_class'].astype(object)
    train['cabin'].fillna('noca', inplace=True)
    train['cabin'] = train['cabin'].apply(lambda x: list(str(x))[0])
    del train['ticket']
    test['t_class'] = ""
    test['t_no'] = ""
    for i in range(len(test['ticket'])):
        words = str(test['ticket'].iloc[i]).split(" ")
        if words[-1].isdigit():
            test['t_no'].iloc[i] = words[-1]
        else:
            test['t_no'].iloc[i] = 0
        if len(words) == 1:
            test['t_class'].iloc[i] = 'no'
        else:
            test['t_class'].iloc[i] = re.sub("[^a-zA-Z0-9]", " ", "".join(words[:-1]))
    test['t_no'] = test['t_no'].astype(float)
    test['t_class'] = test['t_class'].astype(object)
    test['cabin'].fillna('noca', inplace=True)
    test['cabin'] = test['cabin'].apply(lambda x: list(str(x))[0])
    del test['ticket']
    train.reindex(np.random.permutation(train.index))
    y = train['survived']
    x = train.ix[:, 1:]
    for col in list(test):
        if np.issubdtype(test[col].dtype, np.number):
            test[col].fillna(test[col].median(), inplace=True)
            x[col].fillna(x[col].median(), inplace=True)
        else:
            test[col].fillna(test[col].mode()[0], inplace=True)
            x[col].fillna(x[col].mode()[0], inplace=True)
    x = pd.get_dummies(x)
    test = pd.get_dummies(test)
    common_var = list(set(x).intersection(test))
    x = x[common_var]
    test = test[common_var]
    return x,y,test,common_var


def tune_rf(dataset, depth,size,number, sample_size, variable_size, var_names):
    training = dataset[:int(0.8 * len(dataset))]
    valx = [row[:-1] for row in dataset[int(0.8 * len(dataset)):]]
    valy = [row[-1] for row in dataset[int(0.8 * len(dataset)):]]
    score=dict()
    for n in range(len(number)):
        for d in range(len(depth)):
            for s in range(len(size)):
                model = random_forest(training, depth[d], size[s], number[n], sample_size, variable_size, var_names)
                pred = randf_predict(model,valx)
                score[str(n)+str(d)+str(s)]=one_zero_loss(valy, pred)
    print sorted(score.items(), key=lambda x: x[1], reverse=True)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":

    ###### census #####
    x, y, test, vars = load_census()
    data = pd.concat([x, y], axis=1).values.tolist()
    training = data[:int(0.8 * len(data))]
    trainx = [row[:-1] for row in training]
    trainy = [row[-1] for row in training]
    valx = [row[:-1] for row in data[int(0.8 * len(data)):]]
    valy = [row[-1] for row in data[int(0.8 * len(data)):]]

    tree = decision_tree(data[:3000],20,10)
    # a. print predict path
    single_predict_path(tree,x.values.tolist()[0],vars)
    # b. find the most common root split
    a,b=random_forest(data,1,10,300,1000,0.1,vars)
    print b

    tree= decision_tree(training,max_depth=20,min_size=10)
    predtrain = predict(tree,trainx)
    print one_zero_loss(trainy,predtrain)
    predval = predict(tree,valx)
    print one_zero_loss(valy,predval)
    #0.95525
    #0.825
    #0.8364
    #0.8336

    forest, roots = random_forest(training, 10, 10, 100, 2000, 0.5, vars)
    predtrain = randf_predict(forest, trainx)
    print one_zero_loss(trainy, predtrain)
    predval = randf_predict(forest, valx)
    print one_zero_loss(valy, predval)


    x, y, test, vars = load_census()
    data = pd.concat([x, y], axis=1)
    data.reindex(np.random.permutation(data.index))
    data=data.values.tolist()[:2000]
    training = data[:int(0.8*len(data))]
    valx=[row[:-1] for row in data[int(0.8*len(data)):]]
    valy=[row[-1] for row in data[int(0.8*len(data)):]]
    loss=list()
    for i in range(41):
        tree = decision_tree(training,i,10)
        pred = predict(tree,valx)
        loss.append(one_zero_loss(valy,pred))

    plt.plot(range(41), loss)
    plt.title("Census")
    plt.xlabel('Depth of tree')
    plt.ylabel('Validation Accuracy')
    plt.show()

    number=[50,80,100,200,300,400,500]
    sample_size=[1000,2000,3000,4000,5000]
    depth=[3,5,7,10,15,20,30,40]
    size=20
    variable_size=0.8
    tune_rf(data, depth, size,number, sample_size, variable_size, vars)

    ###### titanic ####

    x, y, test, vars = load_titanic()
    position = np.random.permutation(x.shape[0]).tolist()
    data = pd.concat([x, y], axis=1).values.tolist()
    training = data[:int(0.8 * len(data))]
    trainx = [row[:-1] for row in training]
    trainy = [row[-1] for row in training]
    valx = [row[:-1] for row in data[int(0.8 * len(data)):]]
    valy = [row[-1] for row in data[int(0.8 * len(data)):]]

    tree = decision_tree(training, max_depth=50, min_size=10)
    predtrain = predict(tree, trainx)
    print one_zero_loss(trainy, predtrain)
    predval = predict(tree, valx)
    print one_zero_loss(valy, predval)

    forest,roots= random_forest(training,10,10,100,2000,0.5,vars)
    predtrain = randf_predict(forest,trainx)
    print one_zero_loss(trainy,predtrain)
    predval = randf_predict(forest,valx)
    print one_zero_loss(valy,predval)

    tree = decision_tree(data[:5000], 1, 10)
    print_tree(tree,vars,1)

    number=[50,80,100,200,300,400,500]
    sample_size=[1000,2000,3000,4000,5000]
    depth=[3,5,7,10,15,20,30,40]
    size=20
    variable_size=0.8
    tune_rf(data, depth, size,number, sample_size, variable_size, vars)
    """
    depth: 1 Split variable sex_male Split value: 1.0
    left
    depth: 2 Split variable pclass Split value: 3.0
    left
    depth: 3 Split variable t_no Split value: 237249.0
    0.0
    0.0
    right
    depth: 3 Split variable fare Split value: 23.45
    0.0
    0.0
    right
    depth: 2 Split variable cabin_no Split value: 1.0
    left
    depth: 3 Split variable age Split value: 18.0
    1.0
    0.0
    right
    depth: 3 Split variable age Split value: 4.0
    0.0
    0.0
    """
    ###### spam ####
    x, y, test, vars = load_spam(k=50)
    position = np.random.permutation(x.shape[0]).tolist()
    x,y=x[position,:],y[position]
    data = np.concatenate((x, y.reshape((len(y), 1))), axis=1).tolist()[:3000]
    training = data[:int(0.8 * len(data))]
    trainx = [row[:-1] for row in training]
    trainy = [row[-1] for row in training]
    valx = [row[:-1] for row in data[int(0.8 * len(data)):]]
    valy = [row[-1] for row in data[int(0.8 * len(data)):]]


    tree = decision_tree(data, 20, 10)
    # a. print predict path
    single_predict_path(tree, x.tolist()[1], vars)
    # b. find the most common root split
    a, b = random_forest(data, 1, 10, 500, 100, 0.1, vars)
    print b

    tree = decision_tree(training, max_depth=50, min_size=10)
    predtrain = predict(tree, trainx)
    print one_zero_loss(trainy, predtrain)
    predval = predict(tree, valx)
    print one_zero_loss(valy, predval)

    forest,roots= random_forest(training,10,10,100,2000,0.5,vars)
    predtrain = randf_predict(forest,trainx)
    print one_zero_loss(trainy,predtrain)
    predval = randf_predict(forest,valx)
    print one_zero_loss(valy,predval)


    number=[50,80,100,200,300,400,500]
    sample_size=[1000,2000,3000,4000,5000]
    depth=[3,5,7,10,15,20,30,40]
    size=20
    variable_size=0.8
    tune_rf(data, depth, size,number, sample_size, variable_size, vars)


        






