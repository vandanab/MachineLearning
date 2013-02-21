# K-Nearest Neighbor Classifier
'''
Created on Mar 24, 2012
@author: vandana
'''

from utilities import loadiris, loadcars, loadmushroom, loadvoting, loadheartdisease, loadwine, ttestvaluecalculation
from math import sqrt
import operator
import sys
from copy import deepcopy
from pylab import figure, plot, show, xlabel, ylabel, suptitle, text

class NearestNeighbor:
    def __init__(self, k, inputdata, ntgrowth=False, featureselection=False, validationset=None):
        self.k = k
        self.traindict = inputdata.rows
        self.labels = inputdata.labels
        self.attrmap = inputdata.attrmap
        self.numattrs = len(self.attrmap)
        self.ntgrowth = ntgrowth
        self.fs = featureselection
        if validationset:
            self.validationset = validationset
        if ntgrowth:
            self.traindictntgrowth = self.ntgrowthtrain()
        if featureselection:
            self.featureselectionsequence = []
            self.traindictfs = self.sfs_featureselection()
         
    def predict(self, datas, traind=None):
        errors = []
        trainset = self.traindict
        if self.ntgrowth:
            trainset = self.traindictntgrowth
        if self.fs and not traind:
            trainset = self.traindictfs
        if traind:
            trainset = traind
        for i in datas.rows:
            distarr = []
            for j in trainset:
                d = self.dist(i, j)
                distarr.append({'neighbor':j, 'dist':d})
            distarr1 = sorted(distarr, key=lambda x:x['dist'])
            classes = {}
            for j in range(self.k):
                neighbor = distarr1[j]
                if neighbor['neighbor']['target'] in classes:
                    classes[neighbor['neighbor']['target']] += 1
                else:
                    classes[neighbor['neighbor']['target']] = 1
            classes = sorted(classes.iteritems(), key=operator.itemgetter(1), reverse=True)
            majorityclass = classes[0]
            if majorityclass[0] != i['target']:
                errors.append(i)
        acc = (1 - (float(len(errors))/len(datas.rows)))*100
        print 'Accuracy: ', acc
        return acc
    
    def predictdistweighted(self, datas):
        errors = []
        trainset = self.traindict
        if self.ntgrowth:
            trainset = self.traindictntgrowth
        if self.fs:
            trainset = self.traindictfs
        for i in datas.rows:
            #weightarr = []
            weightsum = 0
            classes = {}
            for j in trainset:
                weight = 1
                d = self.dist(i, j)
                if d > 0:
                    weight = 1/(d**2)
                #weightarr.append({'neighbor':j, 'weight':weight})
                weightsum += weight
                if j['target'] in classes:
                    classes[j['target']] += weight
                else:
                    classes[j['target']] = weight
            for j in classes:
                classes[j] = classes[j]/weightsum
            classes = sorted(classes.iteritems(), key=operator.itemgetter(1), reverse=True)
            majorityclass = classes[0]
            if majorityclass[0] != i['target']:
                errors.append(i)
        acc = (1 - (float(len(errors))/len(datas.rows)))*100
        print 'Accuracy: ', acc
        return acc
    
    def dist(self, p1, p2):
        distsq = 0.0
        for i in range(self.numattrs):
            if self.attrmap[i]['typ'] == 'discrete':
                if p1['attrs'][i] == p2['attrs'][i]:
                    attrdist = 0
                else:
                    attrdist = 1
            else:
                attrdist = p1['attrs'][i] - p2['attrs'][i]
            distsq += attrdist**2
        return sqrt(distsq)
    
    def ntgrowthtrain(self):
        ntgrowthtrain = []
        for i in self.traindict:
            mindist = sys.maxint
            nearestneighbor = None
            for j in ntgrowthtrain:
                d = self.dist(i, j)
                if d < mindist:
                    mindist = d
                    nearestneighbor = j
            if nearestneighbor and nearestneighbor['target'] == i['target']:
                continue
            else:
                ntgrowthtrain.append(i)
        print len(ntgrowthtrain), len(self.traindict)
        return ntgrowthtrain
    
    def chisquare_featureweighting(self):
        return ""
    
    def sfs_featureselection(self):
        trainset = self.traindict
        if self.ntgrowth:
            trainset = self.traindictntgrowth
        if not self.validationset:
            print 'pass validation set'
        featureset = [0]* self.numattrs
        features = range(0, self.numattrs)
        acc = 0
        while True:
            (fs, newts, acc, newfs) = self.selectfeature(features, trainset, acc, featureset)
            if acc == 0:
                break;
            else:
                features, newtrainset, featureset = fs, newts, newfs
        print featureset
        """
        print 'Selected Features:',
        for i in range(self.numattrs):
            if featureset[i] == 1:
                print self.attrmap[i]['name'],
        print
        """
        print self.featureselectionsequence
        return newtrainset
    
    def selectfeature(self, features, trainset, prevacc, featureset):
        maxacc = -1
        feature = -1
        maxtrainset = []
        for i in features:
            newtrainset = []
            for j in trainset:
                attrs = [0]*len(j['attrs'])
                for k in range(len(j['attrs'])):
                    attrs[k] = j['attrs'][k]*featureset[k]
                attrs[i] = j['attrs'][i]
                newtrainset.append({'attrs':attrs, 'target':j['target']})
            acc = self.predict(self.validationset, newtrainset)
            if acc > maxacc:
                maxacc = acc
                feature = i
                maxtrainset = deepcopy(newtrainset)
        print maxacc
        if maxacc > prevacc:
            featureset[feature] = 1
            self.featureselectionsequence.append({'feature':self.attrmap[feature]['name'], 'accuracy':maxacc})
            features.remove(feature)
            return (features, maxtrainset, maxacc, featureset)
        return (features, None, 0, None)

def cv10foldresult(k, mldata, ntgrowth=False, fs=False, validationset=None):
    tenfoldcvdata = mldata.crossvalidation10()
    knns = [None] * 10
    accuracies = [0] * 10
    for j in range(0, 10):
        (trdata, tstdata) = mldata.cv10sampledata(tenfoldcvdata, j)
        knns[j] = NearestNeighbor(k, trdata, ntgrowth, fs, validationset)
        acc = knns[j].predict(tstdata)
        #acc = knns[j].predictdistweighted(tstdata)
        #print acc
        accuracies[j] = acc
    meanacc = (sum(accuracies))/10.0
    diffsquares = [(i - meanacc)**2 for i in accuracies]
    sumdiffsquares = sum(diffsquares)
    stderr = sqrt(sumdiffsquares)/10.0
    print accuracies
    print meanacc, stderr
    return "%f%s%f" % (meanacc, u" \u00B1 ", 1.96 * stderr)

def crossvalidate():
    
    ''' Iris dataset '''
    path = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/iris/iris.data'
    mldata = loadiris(path)
    normaldata = mldata.normalizeinput()
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
    normaldata4 = mldata4.normalizeinput()
    ''' Wine dataset '''
    path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
    mldata5 = loadwine(path5)
    normaldata5 = mldata5.normalizeinput()
        
    cvacc = {}
    cvacc['Iris'] = cv10foldresult(3, normaldata)
    cvacc['Cars'] = cv10foldresult(5, mldata1)
    cvacc['Mushrm'] = cv10foldresult(1, mldata2)
    cvacc['Voting'] = cv10foldresult(5, mldata3)
    cvacc['Heart'] = cv10foldresult(5, normaldata4)
    cvacc['Wine'] = cv10foldresult(5, normaldata5)
    
    for i in cvacc:
        print i, ":", cvacc[i]

def bestkexperiment(datasetname, mldata):
    avgaccs = {}
    for k in [1, 3, 5]:
        avg = 0
        for i in range(10):
            sampledmldata = mldata.sampledata({'train':80, 'test':20, 'cv':0})
            knn = NearestNeighbor(k, sampledmldata['train'])
            acc = knn.predict(sampledmldata['test'])
            avg += acc
        avg = avg/10
        avgaccs[k] = avg
    avgaccs = sorted(avgaccs.iteritems(), key=operator.itemgetter(1), reverse=True)
    bestk = avgaccs[0][0]
    print avgaccs[0], datasetname, bestk
    
def main(argv=None):
    
    
    ''' Iris dataset '''
    path = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/iris/iris.data'
    mldata = loadiris(path)
    normaldata = mldata.normalizeinput()
    sampledmldata = normaldata.sampledata({'train':70, 'test':15, 'cv':15})
    #knn = NearestNeighbor(3, sampledmldata['train'])
    #knn = NearestNeighbor(3, sampledmldata['train'], False, True, sampledmldata['cv'])
    knn = NearestNeighbor(3, sampledmldata['train'], True)
    knn.predict(sampledmldata['test'])
    #knn.predict(sampledmldata['test'])
    #knn.predictdistweighted(sampledmldata['test'])
    #bestkexperiment('iris', normaldata)
    
    ''' Cars dataset '''
    path1 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/cars/car.data'
    mldata1 = loadcars(path1)
    sampledmldata1 = mldata1.sampledata({'train':70, 'test':15, 'cv':15})
    #knn1 = NearestNeighbor(5, sampledmldata1['train'])
    #knn1 = NearestNeighbor(5, sampledmldata1['train'], False, True, sampledmldata1['cv'])
    knn1 = NearestNeighbor(5, sampledmldata1['train'], True)
    knn1.predict(sampledmldata1['test'])
    #bestkexperiment('cars', mldata1)
    
    
    ''' Mushroom dataset '''
    path2 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/mushroom/agaricus-lepiota.data'
    mldata2 = loadmushroom(path2)
    sampledmldata2 = mldata2.sampledata({'train':65, 'test':15, 'cv':20})
    #knn2 = NearestNeighbor(1, sampledmldata2['train'])
    #knn2 = NearestNeighbor(1, sampledmldata2['train'], False, True, sampledmldata2['cv'])
    knn2 = NearestNeighbor(1, sampledmldata2['train'], True)
    knn2.predict(sampledmldata2['test'])
    #bestkexperiment('mushroom', mldata2)
    
    ''' Voting dataset '''
    path3 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/voting/house-votes-84.data'
    mldata3 = loadvoting(path3)
    sampledmldata3 = mldata3.sampledata({'train':70, 'test':15, 'cv':15})
    #knn3 = NearestNeighbor(1, sampledmldata3['train'])
    #knn3 = NearestNeighbor(1, sampledmldata3['train'], False, True, sampledmldata3['cv'])
    knn3 = NearestNeighbor(5, sampledmldata3['train'], True)
    knn3.predict(sampledmldata3['test'])
    #bestkexperiment('voting', mldata3)
    
    
    ''' Heart Disease dataset '''
    path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.cleveland.data'
    mldata4 = loadheartdisease(path4, True)
    normaldata4 = mldata4.normalizeinput()
    sampledmldata4 = normaldata4.sampledata({'train':70, 'test':15, 'cv':15})
    #knn4 = NearestNeighbor(5, sampledmldata4['train'])
    #knn4 = NearestNeighbor(5, sampledmldata4['train'], False, True, sampledmldata4['cv'])
    knn4 = NearestNeighbor(5, sampledmldata4['train'], True)
    knn4.predict(sampledmldata4['test'])
    #bestkexperiment('heart disease', normaldata4)
    
    
    ''' Wine dataset '''
    path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
    mldata5 = loadwine(path5)
    normaldata5 = mldata5.normalizeinput()
    sampledmldata5 = normaldata5.sampledata({'train':70, 'test':15, 'cv':15})
    #knn5 = NearestNeighbor(3, sampledmldata5['train'])
    #knn5 = NearestNeighbor(3, sampledmldata5['train'], False, True, sampledmldata5['cv'])
    knn5 = NearestNeighbor(5, sampledmldata5['train'], True)
    knn5.predict(sampledmldata5['test'])
    #bestkexperiment('wine', normaldata5)
    
    
    #crossvalidate()
    #pairedttests()
    #plotdata()

def plotdata():
    sfs = {}
    sfs['iris'] = []
    sfs['iris'].append([{'feature': 'petal_width_cm', 'accuracy': 95.45454545454545}, {'feature': 'petal_len_cm', 'accuracy': 100.0}, {'feature': 'sepal_len_cm', 'accuracy': 100.0}])
    sfs['iris'].append([{'feature': 'petal_width_cm', 'accuracy': 95.45454545454545}, {'feature': 'sepal_len_cm', 'accuracy': 95.45454545454545}])
    sfs['iris'].append([{'feature': 'petal_len_cm', 'accuracy': 90.9090909090909}, {'feature': 'petal_width_cm', 'accuracy': 95.45454545454545}, {'feature': 'sepal_len_cm', 'accuracy': 100.0}])
    
    sfs['cars'] = []
    sfs['cars'].append([{'feature': 'maint', 'accuracy': 66.79536679536679}, {'feature': 'persons', 'accuracy': 68.72586872586872}, {'feature': 'safety', 'accuracy': 74.90347490347492}, {'feature': 'buying', 'accuracy': 84.16988416988417}, {'feature': 'lug_boot', 'accuracy': 91.5057915057915}])
    sfs['cars'].append([{'feature': 'buying', 'accuracy': 66.79536679536679}, {'feature': 'lug_boot', 'accuracy': 66.79536679536679}])
    sfs['cars'].append([{'feature': 'buying', 'accuracy': 71.42857142857143}])
    sfs['cars'].append([{'feature': 'maint', 'accuracy': 69.88416988416988}, {'feature': 'persons', 'accuracy': 71.81467181467181}, {'feature': 'safety', 'accuracy': 77.99227799227799}, {'feature': 'buying', 'accuracy': 86.1003861003861}, {'feature': 'lug_boot', 'accuracy': 90.73359073359073}, {'feature': 'doors', 'accuracy': 90.73359073359073}])
    
    sfs['mushroom'] = []
    sfs['mushroom'].append([{'feature': 'odor', 'accuracy': 98.21428571428571}, {'feature': 'spore-print-color', 'accuracy': 99.32266009852216}, {'feature': 'stalk-color-below-ring', 'accuracy': 99.63054187192118}, {'feature': 'stalk-surface-above-ring', 'accuracy': 100.0}])    
    
    sfs['voting'] = []
    sfs['voting'].append([{'feature': 'physician-fee-freeze', 'accuracy': 98.46153846153847}])
    sfs['voting'].append([{'feature': 'physician-fee-freeze', 'accuracy': 95.38461538461537}, {'feature': 'ex-missile', 'accuracy': 96.92307692307692}])
    sfs['voting'].append([{'feature': 'physician-fee-freeze', 'accuracy': 92.3076923076923}, {'feature': 'ex-missile', 'accuracy': 93.84615384615384}, {'feature': 'education-spending', 'accuracy': 95.38461538461537}])
    sfs['voting'].append([{'feature': 'export-administration-act-sa', 'accuracy': 72.3076923076923}, {'feature': 'physician-fee-freeze', 'accuracy': 98.46153846153847}, {'feature': 'adoption-of-budget-resolution', 'accuracy': 100.0}, {'feature': 'handicapped-infants', 'accuracy': 100.0}, {'feature': 'water-project-cost-sharing', 'accuracy': 100.0}, {'feature': 'el-salvador-aid', 'accuracy': 100.0}, {'feature': 'religious-groups-in-school', 'accuracy': 100.0}, {'feature': 'anti-satellite-test-ban', 'accuracy': 100.0}, {'feature': 'aid-to-nicaraguan-contras', 'accuracy': 100.0}, {'feature': 'synfuels-corporation-cutback', 'accuracy': 100.0}, {'feature': 'education-spending', 'accuracy': 100.0}, {'feature': 'ex-missile', 'accuracy': 100.0}, {'feature': 'crime', 'accuracy': 100.0}])
    
    sfs['heart'] = []
    sfs['heart'].append([{'feature': 'age', 'accuracy': 60.0}, {'feature': 'ca', 'accuracy': 64.44444444444444}, {'feature': 'slope', 'accuracy': 73.33333333333334}])
    sfs['heart'].append([{'feature': 'chol', 'accuracy': 53.333333333333336}, {'feature': 'exang', 'accuracy': 55.55555555555556}, {'feature': 'trestbps', 'accuracy': 62.22222222222222}])
    sfs['heart'].append([{'feature': 'thal', 'accuracy': 51.11111111111111}, {'feature': 'age', 'accuracy': 53.333333333333336}, {'feature': 'slope', 'accuracy': 60.0}, {'feature': 'cp', 'accuracy': 64.44444444444444}, {'feature': 'exang', 'accuracy': 66.66666666666667}])
    sfs['heart'].append([{'feature': 'trestbps', 'accuracy': 62.22222222222222}, {'feature': 'cp', 'accuracy': 64.44444444444444}, {'feature': 'sex', 'accuracy': 66.66666666666667}, {'feature': 'ca', 'accuracy': 68.88888888888889}, {'feature': 'fbs', 'accuracy': 71.11111111111111}])
    
    sfs['wine'] = []
    sfs['wine'].append([{'feature': 'flavanoids', 'accuracy': 76.92307692307692}, {'feature': 'alcohol', 'accuracy': 88.46153846153845}, {'feature': 'proline', 'accuracy': 96.15384615384616}, {'feature': 'color-intensity', 'accuracy': 100.0}])
    sfs['wine'].append([{'feature': 'total-phenols', 'accuracy': 76.92307692307692}, {'feature': 'proline', 'accuracy': 92.3076923076923}, {'feature': 'proanthocyanins', 'accuracy': 100.0}])
    sfs['wine'].append([{'feature': 'flavanoids', 'accuracy': 80.76923076923077}, {'feature': 'alcohol', 'accuracy': 96.15384615384616}, {'feature': 'proline', 'accuracy': 100.0}])
    sfs['wine'].append([{'feature': 'flavanoids', 'accuracy': 80.76923076923077}, {'feature': 'proline', 'accuracy': 96.15384615384616}, {'feature': 'alcohol', 'accuracy': 100.0}])
    
    c = 0
    for i in sfs:
        figure(c)
        for j in sfs[i]:
            xvalues = []
            yvalues = []
            for k in range(len(j)):
                xvalues.append(j[k]['feature'])
                yvalues.append(j[k]['accuracy'])
            plot(range(len(xvalues)), yvalues)
            for l in range(len(xvalues)):
                text(l, yvalues[l], xvalues[l], fontsize=12)
            xlabel('Features')
            ylabel('Accuracy')
        suptitle(i+' Stepwise-Forward Selection')
        c += 1
    show()
        
    
def pairedttests():
    accuracies = {}
    print 'Paired t-test between k-nearest neighbor and distance weighted nearest neighbor schemes'
    accuracies['iris'] = {'nearestneighbork':{'mean':94.6666666667, 'stddev':1.57762127549}, 'nearestneighbordist':{'mean':94.6666666667, 'stddev':2.63312235442}}
    accuracies['heart'] = {'nearestneighbork':{'mean':59.6363636364, 'stddev':4.03242140728}, 'nearestneighbordist':{'mean':54.1212121212, 'stddev':2.57279687239}}
    accuracies['cars'] = {'nearestneighbork':{'mean':88.5426356589, 'stddev':0.678376517752}, 'nearestneighbordist':{'mean':70.0387596899, 'stddev':0.621485974917}}
    accuracies['mushroom'] = {'nearestneighbork':{'mean':100.00, 'stddev':0.0}, 'nearestneighbordist':{'mean':90.2388196658, 'stddev':0.356439214413}}
    accuracies['wine'] = {'nearestneighbork':{'mean':97.2470588235, 'stddev':1.21583553796}, 'nearestneighbordist':{'mean':97.835294, 'stddev':0.854088477868}}
    accuracies['voting'] = {'nearestneighbork':{'mean':93.769379845, 'stddev':1.32437172808}, 'nearestneighbordist':{'mean':90.0968992248, 'stddev':1.00043933021}}
    for i in accuracies:
        (t, dof) = ttestvaluecalculation(accuracies[i]['nearestneighbork']['mean'], accuracies[i]['nearestneighbork']['stddev'], accuracies[i]['nearestneighbordist']['mean'], accuracies[i]['nearestneighbordist']['stddev'], 10)
        print i, ' t-value:', t, ' dof:', dof
    
    accuracies = {}
    print 'Paired t-test between k-nearest neighbor and neural networks'
    accuracies['iris'] = {'nearestneighbork':{'mean':94.6666666667, 'stddev':1.57762127549}, 'neuralnetworks':{'mean':96.000000, 'stddev':1.932183673}}
    accuracies['heart'] = {'nearestneighbork':{'mean':59.6363636364, 'stddev':4.03242140728}, 'neuralnetworks':{'mean':49.575758, 'stddev':3.162335714}}
    accuracies['cars'] = {'nearestneighbork':{'mean':88.5426356589, 'stddev':0.678376517752}, 'neuralnetworks':{'mean':87.870801, 'stddev':2.353145408}}
    accuracies['mushroom'] = {'nearestneighbork':{'mean':100.00, 'stddev':0.0}, 'neuralnetworks':{'mean':99.963054, 'stddev':0.024936735}}
    accuracies['wine'] = {'nearestneighbork':{'mean':97.2470588235, 'stddev':1.21583553796}, 'neuralnetworks':{'mean':97.647059, 'stddev':0.911290306}}
    accuracies['voting'] = {'nearestneighbork':{'mean':93.769379845, 'stddev':1.32437172808}, 'neuralnetworks':{'mean':95.421512, 'stddev':0.787369388}}
    for i in accuracies:
        (t, dof) = ttestvaluecalculation(accuracies[i]['nearestneighbork']['mean'], accuracies[i]['nearestneighbork']['stddev'], accuracies[i]['neuralnetworks']['mean'], accuracies[i]['neuralnetworks']['stddev'], 10)
        print i, ' t-value:', t, ' dof:', dof
    
    accuracies = {}
    print 'Paired t-test between k-nearest neighbor and decision trees'
    accuracies['iris'] = {'nearestneighbork':{'mean':94.6666666667, 'stddev':1.57762127549}, 'decisiontree':{'mean':94.000000, 'stddev':1.475729592}}
    accuracies['heart'] = {'nearestneighbork':{'mean':59.6363636364, 'stddev':4.03242140728}, 'decisiontree':{'mean':52.484848, 'stddev':3.048671429}}
    accuracies['cars'] = {'nearestneighbork':{'mean':88.5426356589, 'stddev':0.678376517752}, 'decisiontree':{'mean':94.399225, 'stddev':0.504955612}}
    accuracies['mushroom'] = {'nearestneighbork':{'mean':100.00, 'stddev':0.0}, 'decisiontree':{'mean':99.409048, 'stddev':0.073534694}}
    #accuracies['wine'] = {'nearestneighbork':{'mean':97.2470588235, 'stddev':1.21583553796}, 'decisiontree':{'mean':97.835294, 'stddev':0.854088477868}}
    accuracies['voting'] = {'nearestneighbork':{'mean':93.769379845, 'stddev':1.32437172808}, 'decisiontree':{'mean':94.723837, 'stddev':1.180112755}}
    for i in accuracies:
        (t, dof) = ttestvaluecalculation(accuracies[i]['nearestneighbork']['mean'], accuracies[i]['nearestneighbork']['stddev'], accuracies[i]['decisiontree']['mean'], accuracies[i]['decisiontree']['stddev'], 10)
        print i, ' t-value:', t, ' dof:', dof
    
if __name__ == "__main__":
    sys.exit(main())