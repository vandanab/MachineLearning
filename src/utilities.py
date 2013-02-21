#utility functions
'''
Created on Jan 31, 2012
@author: vandana
'''
import csv
import random
from math import sqrt

class InputData:
    """ input data structure """
    def __init__(self, labels = None, attrmap = None):
        if labels is None:
            self.labels = set();
        else:
            self.labels = labels
        if attrmap is None:
            self.attrmap = []
        else:
            self.attrmap = attrmap
        self.rows = []
        
    """ percent describes the percentages of the train, test, validation sets """
    def sampledata(self, percent):
        traindata = InputData(labels = self.labels, attrmap = self.attrmap)
        testdata = InputData(labels = self.labels, attrmap = self.attrmap)
        validationdata = None
        random.shuffle(self.rows)
        
        size = len(self.rows)
        size_train = int((percent['train']*size)/100)
        size_test = int((percent['test']*size)/100)

        traindata.rows = self.rows[:size_train]
        if percent['cv']:
            validationdata = InputData(labels = self.labels, attrmap = self.attrmap)
            size_cv = int((percent['cv']*size)/100)
            cv_index = size_train+size_cv
            validationdata.rows = self.rows[size_train:cv_index]
            testdata.rows = self.rows[cv_index:cv_index+size_test]
        else:
            testdata.rows = self.rows[size_train:size_train+size_test]
        return {'train':traindata, 'test':testdata, 'cv':validationdata}
    
    def crossvalidation10(self):
        i = 0
        datab = [[]] * 10
        random.shuffle(self.rows)
        size = len(self.rows)
        chunksize = size/10
        prev = 0
        while i < 9:
            datab[i] = self.rows[prev:prev+chunksize] 
            prev += chunksize
            i += 1
        datab[i] = self.rows[prev:]
        return datab

    def cv10sampledata(self, datab, i):
        trdata = InputData(self.labels, self.attrmap)
        tsdata = InputData(self.labels, self.attrmap)
        len1 = len(datab)
        l1 = []
        l2 = []
        for j in range(len1):
            if j != i:
                l1.extend(datab[j])
            else:
                l2.extend(datab[j])
        trdata.rows = l1
        tsdata.rows = l2
        return (trdata, tsdata)

    def normalizeinput(self):
        data1 = self.rows
        numattrs = len(self.attrmap)
        meanvarmap = [{} for i in range(numattrs)]
        for i in range(numattrs):
            meanvarmap[i]['sum'] = 0
            meanvarmap[i]['sumsquarediffs'] = 0
            meanvarmap[i]['mean'] = 0
            meanvarmap[i]['stddev'] = 0
        numexamples = len(data1) #assumption all data is considered, might remove data when we have missing values    
        for i in data1:
            attrs = i['attrs']
            for j in range(numattrs):
                if self.attrmap[j]['typ'] == 'continuous' and attrs[j] != "?":
                    x = float(attrs[j])
                    meanvarmap[j]['sum'] += x
        for i in range(numattrs):
            meanvarmap[i]['mean'] = meanvarmap[i]['sum']/numexamples
        for i in data1:
            attrs = i['attrs']
            for j in range(numattrs):
                if self.attrmap[j]['typ'] == 'continuous' and attrs[j] != "?":
                    x = float(attrs[j])
                    meanvarmap[j]['sumsquarediffs'] += (x - meanvarmap[j]['mean'])**2
        for i in range(numattrs):
            meanvarmap[i]['stddev'] = sqrt(meanvarmap[i]['sumsquarediffs']/numexamples)
        #print meanvarmap
        for i in data1:
            attrs = i['attrs']
            for j in range(numattrs):
                if self.attrmap[j]['typ'] == 'continuous':
                    if attrs[j] == '?' or meanvarmap[j]['stddev'] == 0:
                        attrs[j] = 0
                    else:
                        attrs[j] = (float(attrs[j]) - meanvarmap[j]['mean'])/meanvarmap[j]['stddev']
        return self

def loadiris(path):
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[4]
            dat = {'attrs' : map(float, row[:4]), 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    mldata.attrmap = [{"name":"sepal_len_cm", "typ":"continuous", "index": 0, "val":[]}, {"name":"sepal_width_cm", "typ":"continuous", "index": 1, "val":[]}, {"name":"petal_len_cm", "typ":"continuous", "index": 2, "val":[]}, {"name":"petal_width_cm", "typ":"continuous", "index": 3, "val":[]}]
    return mldata

def loadcars(path):
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[6]
            dat = {'attrs' : row[:6], 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    mldata.attrmap = [{"name":"buying", "typ":"discrete", "index": 0, "val":["vhigh", "high", "med", "low"]}, {"name":"maint", "typ":"discrete", "index": 1, "val":["vhigh", "high", "med", "low"]}, {"name":"doors", "typ":"discrete", "index": 2, "val":["2", "3", "4", "5more"]}, {"name":"persons", "typ":"discrete", "index": 3, "val":["2", "4", "more"]}, {"name":"lug_boot", "typ":"discrete", "index": 4, "val":["small", "med", "big"]}, {"name":"safety", "typ":"discrete", "index": 5, "val":["low", "med", "high"]}]
    return mldata

def loadheartdisease(path, nn=False, swit=False):
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[13]
            dat = {'attrs' : row[:13], 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    if nn:
        if swit:
            mldata.attrmap = [{"name":"age", "typ":"continuous", "index": 0, "val":[]}, {"name":"sex", "typ":"discrete", "index": 1, "val":["1", "0"]}, {"name":"cp", "typ":"discrete", "index": 2, "val":["1", "2", "3", "4"]}, {"name":"trestbps", "typ":"continuous", "index": 3, "val":[]}, {"name":"chol", "typ":"continuous", "index": 4, "val":[]}, {"name":"fbs", "typ":"discrete", "index": 5, "val":["1", "0", "?"]}, {"name":"restecg", "typ":"discrete", "index": 6, "val":["0", "1", "2", "?"]}, {"name":"thalach", "typ":"continuous", "index": 7, "val":[]}, {"name":"exang", "typ":"discrete", "index": 8, "val":["1", "0", "?"]}, {"name":"oldpeak", "typ":"continuous", "index": 9, "val":[]}, {"name":"slope", "typ":"discrete", "index": 10, "val":["1", "2", "3", "?"]}, {"name":"ca", "typ":"discrete", "index": 11, "val":["0", "1", "2", "3", "?"]}, {"name":"thal", "typ":"discrete", "index": 12, "val":["3", "6", "7", "?"]}]
        else:
            mldata.attrmap = [{"name":"age", "typ":"continuous", "index": 0, "val":[]}, {"name":"sex", "typ":"discrete", "index": 1, "val":["1.0", "0.0"]}, {"name":"cp", "typ":"discrete", "index": 2, "val":["1.0", "2.0", "3.0", "4.0"]}, {"name":"trestbps", "typ":"continuous", "index": 3, "val":[]}, {"name":"chol", "typ":"continuous", "index": 4, "val":[]}, {"name":"fbs", "typ":"discrete", "index": 5, "val":["1.0", "0.0", "?"]}, {"name":"restecg", "typ":"discrete", "index": 6, "val":["0.0", "1.0", "2.0", "?"]}, {"name":"thalach", "typ":"continuous", "index": 7, "val":[]}, {"name":"exang", "typ":"discrete", "index": 8, "val":["1.0", "0.0", "?"]}, {"name":"oldpeak", "typ":"continuous", "index": 9, "val":[]}, {"name":"slope", "typ":"discrete", "index": 10, "val":["1.0", "2.0", "3.0", "?"]}, {"name":"ca", "typ":"discrete", "index": 11, "val":["0.0", "1.0", "2.0", "3.0", "?"]}, {"name":"thal", "typ":"discrete", "index": 12, "val":["3.0", "6.0", "7.0", "?"]}]
    else:
        mldata.attrmap = [{"name":"age", "typ":"continuous", "index": 0, "val":[]}, {"name":"sex", "typ":"discrete", "index": 1, "val":[1.0, 0.0]}, {"name":"cp", "typ":"discrete", "index": 2, "val":[1.0, 2.0, 3.0, 4.0]}, {"name":"trestbps", "typ":"continuous", "index": 3, "val":[]}, {"name":"chol", "typ":"continuous", "index": 4, "val":[]}, {"name":"fbs", "typ":"discrete", "index": 5, "val":[1.0, 0.0]}, {"name":"restecg", "typ":"discrete", "index": 6, "val":[0.0, 1.0, 2.0]}, {"name":"thalach", "typ":"continuous", "index": 7, "val":[]}, {"name":"exang", "typ":"discrete", "index": 8, "val":[1.0, 0.0]}, {"name":"oldpeak", "typ":"continuous", "index": 9, "val":[]}, {"name":"slope", "typ":"discrete", "index": 10, "val":[1.0, 2.0, 3.0]}, {"name":"ca", "typ":"discrete", "index": 11, "val":[0.0, 1.0, 2.0, 3.0]}, {"name":"thal", "typ":"discrete", "index": 12, "val":[3.0, 6.0, 7.0]}]
    return mldata

def loadvoting(path):
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[0]
            dat = {'attrs' : row[1:], 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    mldata.attrmap = [{"name":"handicapped-infants", "typ":"discrete", "index": 0, "val":["y", "n", "?"]}, {"name":"water-project-cost-sharing", "typ":"discrete", "index": 1, "val":["y", "n", "?"]}, {"name":"adoption-of-budget-resolution", "typ":"discrete", "index": 2, "val":["y", "n", "?"]}, {"name":"physician-fee-freeze", "typ":"discrete", "index": 3, "val":["y", "n", "?"]}, {"name":"el-salvador-aid", "typ":"discrete", "index": 4, "val":["y", "n", "?"]}, {"name":"religious-groups-in-school", "typ":"discrete", "index": 5, "val":["y", "n", "?"]}, {"name":"anti-satellite-test-ban", "typ":"discrete", "index": 6, "val":["y", "n", "?"]}, {"name":"aid-to-nicaraguan-contras", "typ":"discrete", "index": 7, "val":["y", "n", "?"]}, {"name":"ex-missile", "typ":"discrete", "index": 8, "val":["y", "n", "?"]}, {"name":"immigration", "typ":"discrete", "index": 9, "val":["y", "n", "?"]}, {"name":"synfuels-corporation-cutback", "typ":"discrete", "index": 10, "val":["y", "n", "?"]}, {"name":"education-spending", "typ":"discrete", "index": 11, "val":["y", "n", "?"]}, {"name":"super-fund-right-to-sue", "typ":"discrete", "index": 12, "val":["y", "n", "?"]}, {"name":"crime", "typ":"discrete", "index": 13, "val":["y", "n", "?"]}, {"name":"duty-free-exports", "typ":"discrete", "index": 14, "val":["y", "n", "?"]}, {"name":"export-administration-act-sa", "typ":"discrete", "index": 15, "val":["y", "n", "?"]}]
    return mldata

def loadmushroom(path):
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[0]
            dat = {'attrs' : row[1:], 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    mldata.attrmap = [{"name":"cap-shape", "typ":"discrete", "index": 0, "val":["b", "c", "x", "f", "k", "s"]}, {"name":"cap-surface", "typ":"discrete", "index": 1, "val":["f", "g", "y", "s"]}, {"name":"cap-color", "typ":"discrete", "index": 2, "val":["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"]}, {"name":"bruises", "typ":"discrete", "index": 3, "val":["t", "f"]}, {"name":"odor", "typ":"discrete", "index": 4, "val":["a", "l", "c", "y", "f", "m", "n", "p", "s"]}, {"name":"gill-attachment", "typ":"discrete", "index": 5, "val":["a", "d", "f", "n"]}, {"name":"gill-spacing", "typ":"discrete", "index": 6, "val":["c", "w", "d"]}, {"name":"gill-size", "typ":"discrete", "index": 7, "val":["b", "n"]}, {"name":"gill-color", "typ":"discrete", "index": 8, "val":["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"]}, {"name":"stalk-shape", "typ":"discrete", "index": 9, "val":["e", "t"]}, {"name":"stalk-root", "typ":"discrete", "index": 10, "val":["b", "c", "u", "e", "z", "r", "?"]}, {"name":"stalk-surface-above-ring", "typ":"discrete", "index": 11, "val":["f", "y", "k", "s"]}, {"name":"stalk-surface-below-ring", "typ":"discrete", "index": 12, "val":["f", "y", "k", "s"]}, {"name":"stalk-color-above-ring", "typ":"discrete", "index": 13, "val":["n", "b", "c", "g", "o", "p", "e", "w", "y"]}, {"name":"stalk-color-below-ring", "typ":"discrete", "index": 14, "val":["n", "b", "c", "g", "o", "p", "e", "w", "y"]}, {"name":"veil-type", "typ":"discrete", "index": 15, "val":["p", "u"]}, {"name":"veil-color", "typ":"discrete", "index": 16, "val":["n", "o", "w", "y"]}, {"name":"ring-number", "typ":"discrete", "index": 17, "val":["n", "o", "t"]}, {"name":"ring-type", "typ":"discrete", "index": 18, "val":["c", "e", "f", "l", "n", "p", "s", "z"]}, {"name":"spore-print-color", "typ":"discrete", "index": 19, "val":["k", "n", "b", "h", "r", "o", "u", "w", "y"]}, {"name":"population", "typ":"discrete", "index": 20, "val":["a", "c", "n", "s", "v", "y"]}, {"name":"habitat", "typ":"discrete", "index": 21, "val":["g", "l", "m", "p", "u", "w", "d"]}]
    return mldata

def loadwine(path):
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[0]
            dat = {'attrs' : row[1:], 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    mldata.attrmap = [{"name":"alcohol", "typ":"continuous", "index": 0, "val":[]}, {"name":"malic-acid", "typ":"continuous", "index": 1, "val":[]}, {"name":"ash", "typ":"continuous", "index": 2, "val":[]}, {"name":"alcalinity-of-ash", "typ":"continuous", "index": 3, "val":[]}, {"name":"magnesium", "typ":"continuous", "index": 4, "val":[]}, {"name":"total-phenols", "typ":"continuous", "index": 5, "val":[]}, {"name":"flavanoids", "typ":"continuous", "index": 6, "val":[]}, {"name":"nonflavanoid-phenols", "typ":"continuous", "index": 7, "val":[]}, {"name":"proanthocyanins", "typ":"continuous", "index": 8, "val":[]}, {"name":"color-intensity", "typ":"continuous", "index": 9, "val":[]}, {"name":"hue", "typ":"continuous", "index": 10, "val":[]}, {"name":"od280-od315", "typ":"continuous", "index": 11, "val":[]}, {"name":"proline", "typ":"continuous", "index": 12, "val":[]}]
    return mldata

def loadtennis():
    path = "./NeuralNetworks/tennis.data"
    mldata = InputData()
    filecontent = csv.reader(open(path, 'rb'), delimiter = ',')
    for row in filecontent:
        if row:
            target = row[4]
            dat = {'attrs' : row[:4], 'target' : target}
            mldata.rows.append(dat)
            mldata.labels.add(target)
    mldata.attrmap = [{"name":"outlook", "typ":"discrete", "index": 0, "val":["sunny", "rain", "overcast"]}, {"name":"temperature", "typ":"discrete", "index": 1, "val":["hot", "mild", "cool"]}, {"name":"humidity", "typ":"discrete", "index": 2, "val":["high", "normal"]}, {"name":"wind", "typ":"discrete", "index": 3, "val":["weak", "strong"]}]
    return mldata


def loadsanjeevdata():
  path = "./NeuralNetworks/sanjeev.data"
  mldata = InputData()
  f = open(path, 'r')
  filecontent = f.readlines()
  for row in filecontent:
    data = [int(x) for x in row.split()]
    if data:
        target = data[2]
        dat = {'attrs' : data[:2], 'target' : target}
        mldata.rows.append(dat)
        mldata.labels.add(target)
  mldata.attrmap = [{"name":"x", "typ":"discrete", "index": 0, "val":[0, 1, 2, 3, 4]}, {"name":"y", "typ":"discrete", "index": 1, "val":[0, 1, 2, 3, 4]}]
  return mldata  


''' Only for discrete attribute features for now '''
def chisquarecalculation(feature, datas, classes):
    observeddistrib = {}
    expecteddistrib = {}
    numexamplesperfeatureval = {}
    numexamplesperclass = {}
    for i in feature['val']:
        numexamplesperfeatureval[i] = 0
        observeddistrib[i] = {}
        expecteddistrib[i] = {}
    for i in classes:
        numexamplesperclass[i] = 0
    for i in feature['val']:
        for j in classes:
            observeddistrib[i][j] = 0
            expecteddistrib[i][j] = 0
    n = len(datas)
    dof = (len(feature['val'])-1)*(len(classes)-1)
    for i in datas:
        attrs = i['attrs']
        attrval = attrs[feature['index']]
        numexamplesperfeatureval[attrval] += 1
        numexamplesperclass[i['target']] += 1
        observeddistrib[attrval][i['target']] += 1
    for i in feature['val']:
        for j in classes:
            expecteddistrib[i][j] = float(numexamplesperfeatureval[i] * numexamplesperclass[j])/n
    chisquare = 0
    for i in feature['val']:
        for j in classes:
            chisquare += ((observeddistrib[i][j] - expecteddistrib[i][j])**2)/expecteddistrib[i][j]
    return (chisquare, dof)

def ttestvaluecalculation(mean1, sd1, mean2, sd2, n):
    denom = sqrt((sd1**2 + sd2**2)/n)
    t = (mean1 - mean2)/denom
    dof = 2*n - 2
    return (t, dof)
