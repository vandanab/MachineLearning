"""Main method to test the Neural Networks module"""

# Author: Vandana Bachani <vandana.bvj@gmail.com>
# Created on Feb 25, 2012

import sys
import pickle
import operator
from utilities import loadiris, loadcars, loadmushroom, loadvoting, \
    loadheartdisease, loadwine, loadsanjeevdata
from math import sqrt
#from pylab import figure, plot, show, xlabel, ylabel, suptitle
from NeuralNetworks.neural_networks import NeuralNetwork


def main(argv=None):
    
    """
    ''' Iris dataset '''
    path = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/iris/iris.data'
    mldata = loadiris(path)
    normaldata = mldata.normalizeinput()
    sampledmldata = normaldata.sampledata({'train':60, 'test':20, 'cv':20})
    arr = []
    configs = [[3], [], [3,5], [5], [10]]
    for i in configs:
        nn = NeuralNetwork(sampledmldata['train'], sampledmldata['cv'], i, learn_rate=0.05, epoch=700)
        #nn = NeuralNetwork(sampledmldata['train'], sampledmldata['cv'])
        nn.construct_network()
        nn.train_network()
        if hasattr(nn, 'goodone'):
            (acc, mse) = nn.goodone.predict(sampledmldata['test'])
            arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn.stop_epoch})
    print arr
    f = open('expt\iris_c1', 'w')
    pickle.dump(arr, f)
    f.close()
    """
    """
    ''' Cars dataset '''
    path1 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/cars/car.data'
    mldata1 = loadcars(path1)
    sampledmldata1 = mldata1.sampledata({'train':60, 'test':20, 'cv':20})
    arr = []
    configs = [[3], [], [3,5], [5], [10]]
    for i in configs:
        nn1 = NeuralNetwork(sampledmldata1['train'], sampledmldata1['cv'], [5], learn_rate=0.01, epoch=1000, stop_criterion=50)
        nn1.construct_network()
        nn1.train_network()
        if hasattr(nn1, 'goodone'):
            (acc, mse) = nn1.goodone.predict(sampledmldata1['test'])
            arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn1.stop_epoch})
    print arr
    f = open('expt\cars_c1', 'w')
    pickle.dump(arr, f)
    f.close()
    """
    '''cars expt - momentum'''
    """
    path1 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/cars/car.data'
    mldata1 = loadcars(path1)
    sampledmldata1 = mldata1.sampledata({'train':60, 'test':20, 'cv':20})
    nn1 = NeuralNetwork(sampledmldata1['train'], sampledmldata1['cv'], [5], learn_rate=0.01, epoch=1000, stop_criterion=50)
    nn1.construct_network()
    nn1.train_network()
    if hasattr(nn1, 'goodone'):
        (acc, mse) = nn1.goodone.predict(sampledmldata1['test'])
    print acc, mse, nn1.stop_epoch
    """
    """
    ''' Mushroom dataset '''
    path2 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/mushroom/agaricus-lepiota.data'
    mldata2 = loadmushroom(path2)
    sampledmldata2 = mldata2.sampledata({'train':60, 'test':20, 'cv':20})
    arr = []
    configs = [[3], [], [3,5], [5], [10]]
    for i in configs:
        nn2 = NeuralNetwork(sampledmldata2['train'], sampledmldata2['cv'], i, learn_rate=0.05, epoch=50, stop_criterion=5)
        nn2.construct_network()
        nn2.train_network()
        (acc, mse) = nn2.goodone.predict(sampledmldata2['test'])
        arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn2.stop_epoch})
    print arr
    f = open('expt\mush_c', 'w')
    pickle.dump(arr, f)
    f.close()
    """
    """
    ''' Mushroom expt - momentum '''
    path2 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/mushroom/agaricus-lepiota.data'
    mldata2 = loadmushroom(path2)
    sampledmldata2 = mldata2.sampledata({'train':60, 'test':20, 'cv':20})
    nn2 = NeuralNetwork(sampledmldata2['train'], sampledmldata2['cv'], [5], learn_rate=0.08, epoch=50, stop_criterion=3)
    nn2.construct_network()
    nn2.train_network()
    (acc, mse) = nn2.goodone.predict(sampledmldata2['test'])
    print acc, '\t', mse, '\t', nn2.stop_epoch
    """
    """
    ''' Voting dataset '''
    path3 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/voting/house-votes-84.data'
    mldata3 = loadvoting(path3)
    sampledmldata3 = mldata3.sampledata({'train':60, 'test':20, 'cv':20})
    arr = []
    configs = [[3], [], [3,5], [5], [10]]
    for i in configs:
        nn3 = NeuralNetwork(sampledmldata3['train'], sampledmldata3['cv'], i, learn_rate=0.01, epoch=500)
        nn3.construct_network()
        nn3.train_network()
        (acc, mse) = nn3.goodone.predict(sampledmldata3['test'])
        arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn3.stop_epoch})
    print arr
    f = open('expt\cong_c1', 'w')
    pickle.dump(arr, f)
    f.close()
    """
    """
    ''' Heart Disease dataset '''
    path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.cleveland.data'
    #path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.switzerland.data'
    #mldata4 = loadheartdisease(path4, True, True)
    mldata4 = loadheartdisease(path4, True)
    normaldata4 = mldata4.normalizeinput()
    sampledmldata4 = normaldata4.sampledata({'train':70, 'test':15, 'cv':15})
    arr = []
    configs = [[3], [], [3,5], [5], [10], [20], [10,5]]
    for i in configs:
        nn4 = NeuralNetwork(sampledmldata4['train'], sampledmldata4['cv'], i, learn_rate=0.01, epoch=1000)
        #nn4 = NeuralNetwork(sampledmldata4['train'], sampledmldata4['cv'], [50], learn_rate=0.05)
        nn4.construct_network()
        nn4.train_network()
        nn4.predict(sampledmldata4['test'])
        if hasattr(nn4, 'goodone'):
            (acc, mse) = nn4.goodone.predict(sampledmldata4['test'])
            arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn4.stop_epoch})
    print arr
    f = open('expt\heart_c1', 'w')
    pickle.dump(arr, f)
    f.close()
    """
    """
    ''' Wine dataset '''
    path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
    mldata5 = loadwine(path5)
    normaldata5 = mldata5.normalizeinput()
    sampledmldata5 = normaldata5.sampledata({'train':60, 'test':20, 'cv':20})
    arr = []
    configs = [[3], [], [3,5], [5], [10]]
    for i in configs:
        nn5 = NeuralNetwork(sampledmldata5['train'], sampledmldata5['cv'], i, epoch=1000)
        nn5.construct_network()
        nn5.train_network()
        nn5.predict(sampledmldata5['test']) 
        if hasattr(nn5, 'goodone'):
            (acc, mse) = nn5.goodone.predict(sampledmldata5['test'])
            arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn5.stop_epoch})
    print arr
    f = open('expt\wine_c1', 'w')
    pickle.dump(arr, f)
    f.close()
    """
    
    '''
    Test Sanjeev's dataset
    '''
    mldata5 = loadsanjeevdata()
    normaldata5 = mldata5.normalizeinput()
    sampledmldata5 = normaldata5.sampledata({'train':80, 'test':10, 'cv':10})
    arr = []
    configs = [[2], [], [1,2], [2,2]]
    for i in configs:
        nn5 = NeuralNetwork(sampledmldata5['train'], sampledmldata5['cv'], i, epoch=1000)
        nn5.construct_network()
        nn5.train_network()
        nn5.predict(sampledmldata5['test']) 
        if hasattr(nn5, 'goodone'):
            (acc, mse) = nn5.goodone.predict(sampledmldata5['test'])
            arr.append({'config':i, 'accuracy':acc, 'mse':mse, 'stop':nn5.stop_epoch})
    print arr
    
    ''' The analysis functions '''
    #crossvalidate()
    #cvresult()
    #convergence()
    #convergence_table()
    #plotmsedata()
    #hiddennodes()

def hiddennodes():
    print 'Accuracies for different network configurations'
    print '--------------------Iris---------------------'
    f = open('expt\iris_c', 'r')
    f1 = open('expt\iris_c1', 'r')
    r = pickle.load(f)
    r1 = pickle.load(f1)
    hiddennodesdataset(r, r1)
    f.close()
    f1.close()
    print
    print '--------------------Cars---------------------'
    f = open('expt\cars_c', 'r')
    f1 = open('expt\cars_c1', 'r')
    r = pickle.load(f)
    r1 = pickle.load(f1)
    hiddennodesdataset(r, r1)
    f.close()
    f1.close()
    print
    print '--------------------Mushroom---------------------'
    f = open('expt\mush_c', 'r')
    r = pickle.load(f)
    r1 = r
    hiddennodesdataset(r, r1)
    f.close()
    f1.close()
    print
    print '--------------------Heart Disease---------------------'
    f = open('expt\heart_c', 'r')
    f1 = open('expt\heart_c1', 'r')
    r = pickle.load(f)
    r1 = pickle.load(f1)
    hiddennodesdataset(r, r1)
    f.close()
    f1.close()
    print
    print '--------------------Voting---------------------'
    f = open('expt\cong_c', 'r')
    f1 = open('expt\cong_c1', 'r')
    r = pickle.load(f)
    r1 = pickle.load(f1)
    hiddennodesdataset(r, r1)
    f.close()
    f1.close()
    print
    print '--------------------Wine---------------------'
    f = open('expt\wine_c', 'r')
    f1 = open('expt\wine_c1', 'r')
    r = pickle.load(f)
    r1 = pickle.load(f1)
    hiddennodesdataset(r, r1)
    f.close()
    f1.close()

def hiddennodesdataset(l1,l2):
    print 'Config\tAccuracy\tMSE\tStop'
    for i in range(len(l1)):
        if l1[i]['accuracy'] > l2[i]['accuracy']:
            print l1[i]['config'], '\t', round(l1[i]['accuracy'], 4), '\t', l1[i]['mse'], '\t', l1[i]['stop']
        else:
            print l2[i]['config'], '\t', round(l2[i]['accuracy'], 4), '\t', l2[i]['mse'], '\t', l2[i]['stop']

"""        
def plotmsedata():
    f = open("cars_mse", 'r')
    cars = pickle.load(f)
    cars_mse = [x['mse'] for x in cars]
    f.close()
    f = open('heart_mse1', 'r')
    heart = pickle.load(f)
    heart_mse = [x['mse'] for x in heart]
    f.close()
    f = open('wine_mse1', 'r')
    wine = pickle.load(f)
    wine_mse = [x['mse'] for x in wine]
    f.close()
    figure(0)
    plot(range(len(cars)), cars_mse)
    xlabel('No. of Epochs')
    ylabel('Mean Square Error')
    suptitle('Mean Square Error curve for Validation Set for Cars dataset')
    figure(1)
    plot(range(len(heart)), heart_mse)
    xlabel('No. of Epochs')
    ylabel('Mean Square Error')
    suptitle('Mean Square Error curve for Validation Set for Heart Disease dataset')
    figure(2)
    plot(range(len(wine)), wine_mse)
    xlabel('No. of Epochs')
    ylabel('Mean Square Error')
    suptitle('Mean Square Error curve for Validation Set for Wine dataset')
    show()
"""

def convergence():
    arr = []
    f = open('wine_stop', 'w')
    for i in range(10):
        ''' Wine dataset '''
        path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
        mldata5 = loadwine(path5)
        normaldata5 = mldata5.normalizeinput()
        sampledmldata5 = normaldata5.sampledata({'train':60, 'test':20, 'cv':20})
        nn5 = NeuralNetwork(sampledmldata5['train'], sampledmldata5['cv'], [5])
        nn5.construct_network()
        nn5.train_network()
        nn5.predict(sampledmldata5['test']) 
        if hasattr(nn5, 'goodone'):
            (acc, mse) = nn5.goodone.predict(sampledmldata5['test'])
        arr.append({"accuracy":acc, "stop":nn5.stop_epoch, "mse":mse})
    pickle.dump(arr, f)
    f.close()

def convergence_table():
    print 'dataset\t\tStopCrit\tIter1\t\tIter2\t\tIter3\t\tIter4\t\tIter5'
    print '\t\t\t\tAcc|Stop\tAcc|Stop\tAcc|Stop\tAcc|Stop\tAcc|Stop',
    pick('iris')
    pick('cars')
    pick('mushroom')
    pick('voting')
    pick('heart')
    pick('wine')

def pick(set1):
    f = open(set1 + '_stop', 'r')
    arr = pickle.load(f)
    #print arr
    arr = sorted(arr, key=operator.itemgetter('accuracy'), reverse = True)
    sc = 20
    if set1 == 'mushroom':
        sc = 5
    elif set1 == 'heart' or set1 == 'voting' or set1 == 'cars': 
        sc = 100
    print
    print set1, '\t\t', sc, '\t\t',
    for i in range(5):
        stop = arr[i]['stop']
        acc = arr[i]['accuracy']
        print round(acc,2), '|', stop, '\t',
    f.close()

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
    #path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.switzerland.data'
    #mldata4 = loadheartdisease(path4, True, True)
    mldata4 = loadheartdisease(path4, True)
    normaldata4 = mldata4.normalizeinput()
    ''' Wine dataset '''
    path5 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/wine/wine.data'
    mldata5 = loadwine(path5)
    normaldata5 = mldata5.normalizeinput()
        
    cvacc = {}
    """
    cvacc['Iris'] = cv10foldresult(normaldata, [3], learn_rate=0.05, epoch=700, stop_criterion=50)
    cvacc['Cars'] = cv10foldresult(mldata1, [5], learn_rate=0.01, epoch=1000, stop_criterion=100)
    cvacc['Mushrm'] = cv10foldresult(mldata2, [5], learn_rate=0.05, epoch=50, stop_criterion=4)
    cvacc['Voting'] = cv10foldresult(mldata3, [5], learn_rate=0.01, epoch=500, stop_criterion=100)
    cvacc['Heart'] = cv10foldresult(normaldata4, [10, 5], learn_rate=0.01, epoch=1000, stop_criterion=100)
    cvacc['Wine'] = cv10foldresult(normaldata5, [5], learn_rate=0.005, epoch=500, stop_criterion=50)
    """
    
    #cvacc['Iris'] = cv10foldresult(normaldata, None, learn_rate=0.05, epoch=700, stop_criterion=50)
    cvacc['Cars'] = cv10foldresult(mldata1, None, learn_rate=0.01, epoch=1000, stop_criterion=50)
    cvacc['Mushrm'] = cv10foldresult(mldata2, None, learn_rate=0.08, epoch=50, stop_criterion=2)
    #cvacc['Voting'] = cv10foldresult(mldata3, None, learn_rate=0.05, epoch=500, stop_criterion=100)
    #cvacc['Heart'] = cv10foldresult(normaldata4, None, learn_rate=0.01, epoch=1000, stop_criterion=100)
    #cvacc['Wine'] = cv10foldresult(normaldata5, None, learn_rate=0.005, epoch=500, stop_criterion=50)
    
    for i in cvacc:
        print i, ":", cvacc[i]
    f = open("nn_cv", 'w')
    pickle.dump(cvacc, f)
    f.close()

def cvresult():
    f = open("nn_cv", 'r')
    cv = pickle.load(f)
    for i in cv:
        print i
    f.close()

def cv10foldresult(mldata, hiddennodelist, learn_rate, epoch, stop_criterion):
    tenfoldcvdata = mldata.crossvalidation10()
    nns = [None] * 10
    accuracies = [0] * 10
    ms = [0] * 10
    epochs = [0] * 10
    for j in range(0, 10):
        (trdata, tstdata) = mldata.cv10sampledata(tenfoldcvdata, j)
        r1 = NeuralNetwork(trdata, tstdata, hiddennodelist, learn_rate, epoch, stop_criterion)
        r1.construct_network()
        r1.train_network()
        r2 = NeuralNetwork(trdata, tstdata, hiddennodelist, learn_rate, epoch, stop_criterion)
        r2.construct_network()
        r2.train_network()
        (acc1, mse1) = r1.goodone.predict(tstdata)
        (acc2, mse2) = r2.goodone.predict(tstdata)
        if acc1 > acc2:
            (acc, mse) = (acc1, mse1)
            nns[j] = r1
        else:
            (acc, mse) = (acc2, mse2)
            nns[j] = r2
        #nns[j] = NeuralNetwork(trdata, tstdata, hiddennodelist, learn_rate, epoch, stop_criterion)
        #nns[j].construct_network()
        #nns[j].train_network()
        #(acc, mse) = nns[j].goodone.predict(tstdata)
        print acc
        accuracies[j] = acc
        ms[j] = mse
        epochs[j] = nns[j].stop_epoch
    meanacc = (sum(accuracies))/10.0
    diffsquares = [(i - meanacc)**2 for i in accuracies]
    sumdiffsquares = sum(diffsquares)
    stderr = sqrt(sumdiffsquares)/10.0
    print accuracies
    print ms
    print epochs
    return "%f%s%f" % (meanacc, u" \u00B1 ", 1.96 * stderr)
    
if __name__ == "__main__":
    sys.exit(main())
