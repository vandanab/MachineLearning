#Naive Bayes Classifier
'''
Created on Apr 14, 2012
@author: vandana
'''
import sys
import math
from utilities import loadiris, loadcars, loadmushroom, loadvoting, loadheartdisease, loadwine, ttestvaluecalculation

class NaiveBayes:
    def __init__(self, inputdata):
        self.datas = inputdata.rows
        self.classes = inputdata.labels
        self.attrmap = inputdata.attrmap
        self.classpriors = {}
        self.attrprobabilities = {}
        self.getclasswisedata()
        self.numexamples = len(inputdata.rows)
    
    def getclasswisedata(self):
        self.classwisedata = {}
        for i in self.classes:
            self.classwisedata[i] = []
        for i in self.datas:
            self.classwisedata[i['target']].append(i)
    
    def calculateclasspriors(self):
        for i in self.classes:
            self.classpriors[i] = float(len(self.classwisedata[i]))/self.numexamples
        #print self.classpriors
    
    def calculateattrprobabilities(self):
        for i in self.classes:
            self.attrprobabilities[i] = {}
            for j in self.attrmap:
                self.attrprobabilities[i][j['name']] = {}
        for i in self.attrmap:
            index = i['index']
            if i['typ'] == 'discrete':
                for j in self.classwisedata:
                    numinstances = {}
                    for k in i['val']:
                        numinstances[k] = 0
                    for l in self.classwisedata[j]:
                        attrval = l['attrs'][index]
                        numinstances[attrval] += 1
                    for k in i['val']:
                        self.attrprobabilities[j][i['name']][k] = float(numinstances[k])/len(self.classwisedata[j])
            else:
                for j in self.classwisedata:
                    n = len(self.classwisedata[j])
                    maxlikelihoodattrmean = 0
                    sumsquares = 0
                    maxlikelihoodvar = 0
                    for k in self.classwisedata[j]:
                        val = 0
                        if k['attrs'][index] != "?":
                            val = float(k['attrs'][index])
                        maxlikelihoodattrmean += val
                    maxlikelihoodattrmean = maxlikelihoodattrmean/n
                    for k in self.classwisedata[j]:
                        val = 0
                        if k['attrs'][index] == "?":
                            val = maxlikelihoodattrmean
                        else:
                            val = float(k['attrs'][index]) 
                        sumsquares += (val - maxlikelihoodattrmean)**2
                    if n >= 2:
                        maxlikelihoodvar = sumsquares/(n-1)
                    else:
                        maxlikelihoodvar = sumsquares
                    self.attrprobabilities[j][i['name']]['mean'] = maxlikelihoodattrmean
                    self.attrprobabilities[j][i['name']]['variance'] = maxlikelihoodvar
        #print self.attrprobabilities
    
    def train(self):
        self.calculateclasspriors()
        self.calculateattrprobabilities()
        
    def predict(self, testdata):
        errors = []
        for i in testdata.rows:
            attrs = i['attrs']
            maxprob = -1
            maxclass = ""
            for j in self.classes:
                p = self.getprobattrsgivenclass(attrs, j)
                prob = 1
                for k in p:
                    prob *= k
                maxaposteriori = self.classpriors[j] * prob
                if maxaposteriori > maxprob:
                    maxprob = maxaposteriori
                    maxclass = j
            if maxclass != i['target']:
                errors.append(i)
        acc = (1 - float(len(errors))/len(testdata.rows))*100
        print 'Accuracy:', acc
        return acc
    
    def getprobattrsgivenclass(self, attrs, theclass):
        attrprobs = self.attrprobabilities[theclass]
        probs = []
        for i in self.attrmap:
            if i['typ'] == 'discrete':
                attrval = attrs[i['index']]
                probs.append(attrprobs[i['name']][attrval])
            else:
                attrval = attrs[i['index']]
                if attrval == "?":
                    attrval = attrprobs[i['name']]['mean']
                else:
                    attrval = float(attrval)
                prob = calculategaussianprob(attrval, attrprobs[i['name']]['mean'], attrprobs[i['name']]['variance'])
                probs.append(prob)                                                                        
        return probs

def calculategaussianprob(val, mean, variance):
    constant = 1/math.sqrt(2*math.pi*variance)
    expterm = -((val - mean)**2)/(2*variance)
    p = constant * math.exp(expterm)
    return p
        
def main(argv=None):
    
    """
    ''' Iris dataset '''
    path = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/iris/iris.data'
    mldata = loadiris(path)
    #normaldata = mldata.normalizeinput()
    #sampledmldata = normaldata.sampledata({'train':70, 'test':15, 'cv':15})
    sampledmldata = mldata.sampledata({'train':80, 'test':20, 'cv':0})
    nb = NaiveBayes(sampledmldata['train'])
    nb.train()
    nb.predict(sampledmldata['test'])
    
    
    ''' Cars dataset '''
    path1 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/cars/car.data'
    mldata1 = loadcars(path1)
    sampledmldata1 = mldata1.sampledata({'train':70, 'test':15, 'cv':15})
    nb1 = NaiveBayes(sampledmldata1['train'])
    nb1.train()
    nb1.predict(sampledmldata1['test'])
    
    
    ''' Mushroom dataset '''
    path2 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/mushroom/agaricus-lepiota.data'
    mldata2 = loadmushroom(path2)
    sampledmldata2 = mldata2.sampledata({'train':80, 'test':20, 'cv':0})
    nb2 = NaiveBayes(sampledmldata2['train'])
    nb2.train()
    nb2.predict(sampledmldata2['test'])
    
    
    ''' Voting dataset '''
    path3 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/voting/house-votes-84.data'
    mldata3 = loadvoting(path3)
    sampledmldata3 = mldata3.sampledata({'train':80, 'test':20, 'cv':0})
    nb3 = NaiveBayes(sampledmldata3['train'])
    nb3.train()
    nb3.predict(sampledmldata3['test'])
    
    
    ''' Heart Disease dataset '''
    path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.cleveland.data'
    mldata4 = loadheartdisease(path4, True)
    #normaldata4 = mldata4.normalizeinput()
    #sampledmldata4 = normaldata4.sampledata({'train':80, 'test':20, 'cv':0})
    sampledmldata4 = mldata4.sampledata({'train':80, 'test':20, 'cv':0})
    nb4 = NaiveBayes(sampledmldata4['train'])
    nb4.train()
    nb4.predict(sampledmldata4['test'])
    
    
    ''' Wine dataset '''
    path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
    mldata5 = loadwine(path5)
    #normaldata5 = mldata5.normalizeinput()
    #sampledmldata5 = normaldata5.sampledata({'train':80, 'test':20, 'cv':0})
    sampledmldata5 = mldata5.sampledata({'train':80, 'test':20, 'cv':0})
    nb5 = NaiveBayes(sampledmldata5['train'])
    nb5.train()
    nb5.predict(sampledmldata5['test'])
    """
    #crossvalidate()
    pairedttests()

def pairedttests():
    accuracies = {}
    print 'Paired t-test between Naive Bayes and k-Nearest Neighbor'
    accuracies['iris'] = {'nearestneighbork':{'mean':94.6666666667, 'stddev':1.57762127549}, 'naivebayes':{'mean':95.333333, 'stddev':3.227229}}
    accuracies['heart'] = {'nearestneighbork':{'mean':59.6363636364, 'stddev':4.03242140728}, 'naivebayes':{'mean':56.818182, 'stddev':5.402074}}
    accuracies['cars'] = {'nearestneighbork':{'mean':88.5426356589, 'stddev':0.678376517752}, 'naivebayes':{'mean':86.235142, 'stddev':1.230172}}
    accuracies['mushroom'] = {'nearestneighbork':{'mean':100.00, 'stddev':0.0}, 'naivebayes':{'mean':99.643099, 'stddev':0.142510}}
    accuracies['wine'] = {'nearestneighbork':{'mean':97.2470588235, 'stddev':1.21583553796}, 'naivebayes':{'mean':96.258824, 'stddev':3.007528}}
    accuracies['voting'] = {'nearestneighbork':{'mean':93.769379845, 'stddev':1.32437172808}, 'naivebayes':{'mean':90.096899, 'stddev':3.040872}}
    for i in accuracies:
        (t, dof) = ttestvaluecalculation(accuracies[i]['naivebayes']['mean'], accuracies[i]['naivebayes']['stddev'], accuracies[i]['nearestneighbork']['mean'], accuracies[i]['nearestneighbork']['stddev'], 10)
        print i, ' t-value:', t, ' dof:', dof
    
    accuracies = {}
    print 'Paired t-test between Naive Bayes and Neural Networks'
    accuracies['iris'] = {'naivebayes':{'mean':95.333333, 'stddev':3.227229}, 'neuralnetworks':{'mean':96.000000, 'stddev':1.932183673}}
    accuracies['heart'] = {'naivebayes':{'mean':56.818182, 'stddev':5.402074}, 'neuralnetworks':{'mean':49.575758, 'stddev':3.162335714}}
    accuracies['cars'] = {'naivebayes':{'mean':86.235142, 'stddev':1.230172}, 'neuralnetworks':{'mean':87.870801, 'stddev':2.353145408}}
    accuracies['mushroom'] = {'naivebayes':{'mean':99.643099, 'stddev':0.142510}, 'neuralnetworks':{'mean':99.963054, 'stddev':0.024936735}}
    accuracies['wine'] = {'naivebayes':{'mean':96.258824, 'stddev':3.007528}, 'neuralnetworks':{'mean':97.647059, 'stddev':0.911290306}}
    accuracies['voting'] = {'naivebayes':{'mean':90.096899, 'stddev':3.040872}, 'neuralnetworks':{'mean':95.421512, 'stddev':0.787369388}}
    for i in accuracies:
        (t, dof) = ttestvaluecalculation(accuracies[i]['naivebayes']['mean'], accuracies[i]['naivebayes']['stddev'], accuracies[i]['neuralnetworks']['mean'], accuracies[i]['neuralnetworks']['stddev'], 10)
        print i, ' t-value:', t, ' dof:', dof
    
    accuracies = {}
    print 'Paired t-test between Naive Bayes and Decision Trees'
    accuracies['iris'] = {'naivebayes':{'mean':95.333333, 'stddev':3.227229}, 'decisiontree':{'mean':94.000000, 'stddev':1.475729592}}
    accuracies['heart'] = {'naivebayes':{'mean':56.818182, 'stddev':5.402074}, 'decisiontree':{'mean':52.484848, 'stddev':3.048671429}}
    accuracies['cars'] = {'naivebayes':{'mean':86.235142, 'stddev':1.230172}, 'decisiontree':{'mean':94.399225, 'stddev':0.504955612}}
    accuracies['mushroom'] = {'naivebayes':{'mean':99.643099, 'stddev':0.142510}, 'decisiontree':{'mean':99.409048, 'stddev':0.073534694}}
    #accuracies['wine'] = {'naivebayes':{'mean':96.258824, 'stddev':3.007528}, 'decisiontree':{'mean':97.835294, 'stddev':0.854088477868}}
    accuracies['voting'] = {'naivebayes':{'mean':90.096899, 'stddev':3.040872}, 'decisiontree':{'mean':94.723837, 'stddev':1.180112755}}
    for i in accuracies:
        (t, dof) = ttestvaluecalculation(accuracies[i]['naivebayes']['mean'], accuracies[i]['naivebayes']['stddev'], accuracies[i]['decisiontree']['mean'], accuracies[i]['decisiontree']['stddev'], 10)
        print i, ' t-value:', t, ' dof:', dof

def cv10foldresult(mldata):
    tenfoldcvdata = mldata.crossvalidation10()
    nbs = [None] * 10
    accuracies = [0] * 10
    for j in range(0, 10):
        (trdata, tstdata) = mldata.cv10sampledata(tenfoldcvdata, j)
        nbs[j] = NaiveBayes(trdata)
        nbs[j].train()
        acc = nbs[j].predict(tstdata)
        accuracies[j] = acc
    meanacc = (sum(accuracies))/10.0
    diffsquares = [(i - meanacc)**2 for i in accuracies]
    sumdiffsquares = sum(diffsquares)
    stderr = math.sqrt(sumdiffsquares)/10.0
    print accuracies
    print meanacc, stderr
    return "%f%s%f" % (meanacc, u" \u00B1 ", 1.96 * stderr)

def crossvalidate():
    
    ''' Iris dataset '''
    path = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/iris/iris.data'
    mldata = loadiris(path)
    #normaldata = mldata.normalizeinput()
    ''' Cars dataset '''
    path1 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/cars/car.data'
    mldata1 = loadcars(path1)
    ''' Mushroom dataset '''
    path2 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/mushroom/agaricus-lepiota.data'
    mldata2 = loadmushroom(path2)
    ''' Voting dataset '''
    path3 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/voting/house-votes-84.data'
    mldata3 = loadvoting(path3)
    ''' Heart Disease dataset '''
    path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.cleveland.data'
    mldata4 = loadheartdisease(path4, True)
    #normaldata4 = mldata4.normalizeinput()
    ''' Wine dataset '''
    path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
    mldata5 = loadwine(path5)
    #normaldata5 = mldata5.normalizeinput()
        
    cvacc = {}
    cvacc['Iris'] = cv10foldresult(mldata)
    cvacc['Cars'] = cv10foldresult(mldata1)
    cvacc['Mushrm'] = cv10foldresult(mldata2)
    cvacc['Voting'] = cv10foldresult(mldata3)
    cvacc['Heart'] = cv10foldresult(mldata4)
    cvacc['Wine'] = cv10foldresult(mldata5)
    
    for i in cvacc:
        print i, ":", cvacc[i]

if __name__ == '__main__':
    sys.exit(main())
