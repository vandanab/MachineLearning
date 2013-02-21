# Decision Trees Machine Learning
'''
Created on Jan 31, 2012
@author: vandana
'''
from __future__ import division
from utilities import loadiris, loadcars, InputData, loadtennis, loadmushroom, loadvoting, loadheartdisease
from math import log
from copy import deepcopy
import sys
import operator

class dtree:
    """ Decision Trees Algorithm - Machine Learning """
    root = None
    def __init__(self, traindata):
        self.root = node(traindata, 0)
        
    def generatetree(self, node):
        if node.nodepure():
            #print node.label
            return
        node.splitattribute()
        for nodechild in node.children:
            self.generatetree(nodechild)
            node.size += nodechild.size
            
    def predict(self, testdata, setStats=False):
        testattrs = testdata['attrs'];
        if self.root:
            currnode = self.root
            while True:
                if setStats:
                    currnode.teststats['classes'][testdata['target']] += 1
                    currnode.teststats['total'] += 1
                if len(currnode.children) > 0:
                    splitattr = currnode.splitattr
                    if splitattr['typ'] == "discrete":
                        valtype = type(splitattr['val'][0])
                        testval = testattrs[splitattr['index']]
                        if testval == "?":
                            maxval = -1
                            cl = len(currnode.children)
                            for i in range(0, cl):
                                if currnode.children[i].num > maxval:
                                    maxval = currnode.children[i].num
                                    testval = splitattr['val'][i]
                        if valtype is float:
                            testval = float(testval)
                        elif valtype is int:
                            testval = int(testval)
                        ind = splitattr['val'].index(testval)
                        currnode = currnode.children[ind]
                    else:
                        testval = testattrs[splitattr['index']]
                        if testval == "?":
                            if currnode.children[0].num > currnode.children[1].num:
                                testval = currnode.splitval
                            else:
                                testval = currnode.splitval + 0.5 #some value to make it > splitval
                        if testval <= currnode.splitval:
                            currnode = currnode.children[0]
                        else:
                            currnode = currnode.children[1]
                else:
                    break
        return currnode
    
    def testtree(self, testdata, setStats=False, calcAccuracy=False):
        target = []
        prediction = []
        correct = 0
        for i in testdata.rows:
            target.append(i['target'])
            prediction.append(self.predict(i, setStats).label)
        for x, y in zip(target, prediction):
            #print x, y
            if x == y:
                correct += 1
        acc = (correct/len(target))*100
        if calcAccuracy:
            print "accuracy: ", acc
            print "size: ", self.root.size
        return acc
    
    def recalcsize(self, node):
        l = 0
        l1 = len(node.children)
        if l1 > 0:
            for i in node.children:
                l += self.recalcsize(i)
        else:
            l = 1
        node.size = l
        return l
        
    def prune(self, ch, cvdata, printdtree=False):
        if ch == 1:
            self.testtree(cvdata, True)
            self.reducederrprune(self.root)
            self.recalcsize(self.root)
            if printdtree:
                self.printtree()
            return None
        elif ch == 2:
            prunedtrees = self.costcomplexprune()
            maxacc = -1
            besttree = None
            for i in prunedtrees:
                acc = i.testtree(cvdata, False, False)
                if maxacc < acc:
                    maxacc = acc
                    besttree = i
            if besttree:
                besttree.recalcsize(besttree.root)
                if printdtree:
                    besttree.printtree()
                return besttree
    
    def reducederrprune(self, node):
        if len(node.children) == 0:
            testStats = node.teststats['classes']
            correct = testStats[node.label]
            return (node.teststats['total'] - correct)
        else:
            error = 0
            erroratnode = {}
            maxlabel = ""
            maxi = -1
            for i in node.children:
                error += self.reducederrprune(i)
            for i in node.classes.keys():
                erroratnode[i] = 0
            for i in node.teststats['classes']:
                erroratnode[i] = node.teststats['total'] - node.teststats['classes'][i]
                if node.teststats['classes'][i] > maxi:
                    maxi = node.teststats['classes'][i]
                    maxlabel = i
            minerr = sys.maxint
            for i in erroratnode:
                if minerr > erroratnode[i]:
                    minerr = erroratnode[i]
            if error < minerr:
                return error
            else:
                node.children = {}
                node.label = maxlabel
                return node.teststats['total'] - maxi
            
    def costcomplexprune(self):
        prunedtrees = []
        prunedtrees.append(self)
        nosubtrees = False
        i = 1
        while not nosubtrees:
            cp = deepcopy(prunedtrees[i-1])
            dicta = {"alpha":sys.maxint, "minnode":None}
            dicta = self.calcalpha(cp.root, dicta)
            if not dicta["minnode"]:
                nosubtrees = True
                continue
            dicta["minnode"].children = []
            dicta["minnode"].setmajoritylabel()
            prunedtrees.append(cp)
            i += 1
        return prunedtrees
    
    def calcalpha(self, node, mins):
        alpha = mins["alpha"]
        if node.size > 1:
            maxc = -1
            maxcls = None
            for i in node.classes:
                if node.classes[i] > maxc:
                    maxc = node.classes[i]
                    maxcls = i
            errsum = 0;
            for i in node.classes:
                if i != maxcls:
                    errsum += node.classes[i]
            rt = errsum/(self.root.num)
            leaferrsum = 0
            leaferrsum = self.leaferrorcost(node, leaferrsum)
            leaferrsum = leaferrsum/(self.root.num)
            al = (rt - leaferrsum)/(node.size - 1)
            if alpha > al:
                mins["alpha"] = al
                mins["minnode"] = node
            for i in node.children:
                mins = self.calcalpha(i, mins)
        return mins
    
    def leaferrorcost(self, node, sum1):
        if len(node.children) == 0:
            for i in node.classes:
                if i != node.label:
                    sum1 += node.classes[i]
        else:
            for i in node.children:
                sum1 = self.leaferrorcost(i, sum1)
        return sum1

    def printtree(self, debug=False):
        self.root.printnode(debug)
    
class node:
    """ Node of decision tree """
    def __init__(self, data, level):
        self.children = []
        self.classes = {}
        self.teststats = {"classes":{}, "total":0}
        self.level = level
        self.label = ""
        self.num = len(data.rows)
        self.data = data
        for i in data.labels:
            self.classes[i] = 0
            self.teststats['classes'][i] = 0
        for i in data.rows:
            t = i['target']
            self.classes[t] += 1
        self.splitattr = ""
        self.splitval = 0
        self.size = 0
    
    def createleaf(self, label):
        self.size = 1
        self.label = label
    
    def printnode(self, debug):
        tabs = '\t'*self.level
        print '{0}'.format(tabs),
        if not self.label:
            print '[{0}, '.format(self.level),
        else:
            print '[{0} {1}, '.format(self.level, self.label),
        if self.splitattr:
            print 'X={0}, [ '.format(self.splitattr['name']),
        else:
            print '[ ',
        for i in self.classes:
            print '{0} - {1} '.format(i, self.classes[i]),
        print "]]"
        if debug:
            print self.teststats
        for i in self.children:
            i.printnode(debug)
    
    def setmajoritylabel(self):
        m = -1
        for i in self.classes:
            if m < self.classes[i]:
                m = self.classes[i]
                self.label = i
        self.size = 1
        
    def splitattribute(self):
        minent = sys.maxint
        bestf = None
        val = 0
        for i in self.data.attrmap:
            if i['typ'] == "discrete":
                temp = self.split(i)
                if not temp:
                    continue
                e = self.splitentropy(temp, i['typ'])
                if e < minent:
                    minent = e
                    bestf = i
            else:
                x = self.sortbyattr(i)
                poss_splits = []
                prev = x[0]
                for curr in x:
                    if curr['target'] != prev['target']:
                        ca = curr['attrs']
                        pa = prev['attrs']
                        #print ca, pa
                        poss_splits.append((float(ca[i['index']]) + float(pa[i['index']])) / 2)
                    prev = curr
                #print poss_splits
                for j in poss_splits:
                    temp = self.split2(i, j)
                    if not temp:
                        continue
                    e = self.splitentropy(temp, i['typ'])
                    if e < minent:
                        minent = e
                        bestf = i
                        val = j
        if bestf:
            self.splitattr = bestf
            #print bestf, val
            if(bestf['typ'] == "continuous"):
                self.splitval = val 
            self.createchildren()
        else:
            self.setmajoritylabel()
        #self.printnodestate()
    
    def split(self, attr):
        d = self.data.rows
        attrsplit = {}
        vals = attr['val']
        valtype = type(vals[0])
        for i in vals:
            attrsplit[i] = []
        for i in d:
            attrs = i['attrs']
            attrval = attrs[attr['index']]
            if attrval == "?":
                attrval = self.getmissingvalue(i, attr['index'])
                attrs[attr['index']] = attrval
            if valtype is float:
                attrval = float(attrval)
            elif valtype is int:
                attrval = int(attrval)
            attrsplit[attrval].append(i)
        for i in attrsplit:
            if len(attrsplit[i]) < 1:
                return None
        return attrsplit
    
    def split2(self, attr, splitval):
        d = self.data.rows
        attrsplit = [[], []]
        for i in d:
            attrs = i['attrs']
            if attrs[attr['index']] <= splitval :
                attrsplit[0].append(i)
            else:
                attrsplit[1].append(i)
        if len(attrsplit[0]) == 0 or len(attrsplit[1]) == 0:
            return None
        return attrsplit
    
    def getmissingvalue(self, datarow, attrindex):
        cnt = {}
        for i in self.data.attrmap[attrindex]['val']:
            cnt[i] = 0
        valtype = type(self.data.attrmap[attrindex]['val'][0])
        vals = []
        t = datarow['target']
        for i in self.data.rows:
            val = i['attrs'][attrindex]
            if i['target'] == t and val != "?":
                if valtype is float:
                    val = float(val)
                elif valtype is int:
                    val = int(val)
                vals.append(val)
                cnt[val] += 1
        s_cnt = sorted(cnt.iteritems(), key=operator.itemgetter(1), reverse = True)
        for (i, j) in s_cnt:
            return i
    
    def sortbyattr(self, attr):
        d = self.data.rows
        ind = attr['index']
        x = sorted(d, key=lambda r:r['attrs'][ind])
        return x
    
    def splitentropy(self, data, attrtype):
        nm = self.num
        nmi = {}
        I = 0
        c = 0
        k = len(self.classes)
        nmj = [0] * len(data)
        I1 = [0] * len(data)
        for j in data:
            for i in self.classes:
                nmi[i] = 0
            p = [0] * k
            if(attrtype == "discrete"):
                nmj[c] = len(data[j])
                for i in data[j]:
                    nmi[i['target']] += 1
            else:
                nmj[c] = len(j)
                for i in j:
                    nmi[i['target']] += 1
            for l in nmi:
                p.append(prob(nmi[l], nmj[c]))
            I1[c] = entropy(p)
            I += (nmj[c]/nm)*I1[c]
            c += 1
        return I
    
    def createchildren(self):
        ind = self.splitattr['index']
        childrendata = []
        if self.splitattr['typ'] == "continuous":
            r = 2
        else:
            r = len(self.splitattr['val'])
        for i in range(0,r):
            childrendata.append(InputData(self.data.labels, self.data.attrmap))
        for i in self.data.rows:
            if self.splitattr['typ'] == "continuous":
                if i['attrs'][ind] <= self.splitval:
                    childrendata[0].rows.append(i)
                else:
                    childrendata[1].rows.append(i)
            else:
                vals = self.splitattr['val']
                valtype = type(vals[0])
                val = i['attrs'][ind]
                if valtype is float:
                    val = float(val)
                elif valtype is int:
                    val = int(val) 
                j = vals.index(val)
                childrendata[j].rows.append(i)
        l = self.level
        for i in childrendata:
            self.children.append(node(i, l+1))
    
    def nodepure(self):
        cl = self.classes;
        l = [i for i in cl if cl[i] > 0]
        l1 = len(l)
        if l1 == 1:
            self.createleaf(l[0])
            return 1
        else:
            return 0
        
    def printnodestate(self):
        print self.num, " ", self.level, " ", self.splitattr, " ", self.classes
        for i in self.children:
            i.printnodestate()
        
def prob(ni, n):
    """ ni is the number of examples of a particular class """
    if n > 0:
        return (ni/n)
    else:
        return 0

def entropy(p):
    s = 0
    k = len(p)
    for i in range(0, k):
        if p[i] == 0:
            continue
        else:
            s += p[i]*log(p[i], 2)
    return (-s)

def nodeentropy(node):
    p = []
    for i in node.classes:
        p.append(prob(node.classes[i], node.num))
    return entropy(p)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    #toy dataset
    #mldata = loadtennis()
    #sampledmldata = mldata.sampledata({'train':80, 'test':20, 'cv':0})
    
    ''' Iris dataset '''
    path = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/iris/iris.data'
    mldata = loadiris(path)
    sampledmldata = mldata.sampledata({'train':70, 'test':10, 'cv':20})
    dt = dtree(sampledmldata['train'])
    dt.generatetree(dt.root)
    print "before pruning...."
    dt.printtree()
    dt.testtree(sampledmldata['test'], False, True)
    #print "after pruning...."
    #besttree = dt.prune(2, sampledmldata['cv'], True)
    #besttree.testtree(sampledmldata['test'], False, True)
    #dt.prune(1, sampledmldata['cv'], True)
    #dt.testtree(sampledmldata['test'], False, True)
    
    ''' Cars dataset '''
    path1 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/cars/car.data'
    mldata1 = loadcars(path1)
    sampledmldata1 = mldata1.sampledata({'train':70, 'test':10, 'cv':20})
    dt1 = dtree(sampledmldata1['train'])
    dt1.generatetree(dt1.root)
    dt1.printtree()
    dt1.testtree(sampledmldata1['test'], False, True)
    #print "after pruning...."
    #besttree1 = dt1.prune(2, sampledmldata1['cv'], True)
    #besttree1.testtree(sampledmldata1['test'], False, True)
    #dt1.prune(1, sampledmldata1['cv'], True)
    #dt1.testtree(sampledmldata1['test'], False, True)
    
    ''' Mushroom dataset '''
    path2 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/mushroom/agaricus-lepiota.data'
    mldata2 = loadmushroom(path2)
    sampledmldata2 = mldata2.sampledata({'train':70, 'test':10, 'cv':20})
    dt2 = dtree(sampledmldata2['train'])
    dt2.generatetree(dt2.root)
    print "before pruning...."
    dt2.printtree()
    dt2.testtree(sampledmldata2['test'], False, True)
    #print "after pruning...."
    #besttree2 = dt2.prune(2, sampledmldata2['cv'], True)
    #besttree2.testtree(sampledmldata2['test'], False, True)
    #dt2.prune(1, sampledmldata2['cv'], True)
    #dt2.testtree(sampledmldata2['test'], False, True)
    
    ''' Voting dataset '''
    path3 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/voting/house-votes-84.data'
    mldata3 = loadvoting(path3)
    sampledmldata3 = mldata3.sampledata({'train':70, 'test':10, 'cv':20})
    dt3 = dtree(sampledmldata3['train'])
    dt3.generatetree(dt3.root)
    print "before pruning...."
    dt3.printtree()
    dt3.testtree(sampledmldata3['test'], False, True)
    #print "after pruning...."
    #besttree3 = dt3.prune(2, sampledmldata3['cv'], True)
    #besttree3.testtree(sampledmldata3['test'], False, True)
    #dt3.prune(1, sampledmldata3['cv'], True)
    #dt3.testtree(sampledmldata3['test'], False, True)
    
    ''' Heart Disease dataset '''
    path4 = '/host/Users/vandana/Documents/Grad_Studies/Spring2012/Machine Learning/Projects/heart_disease/processed.cleveland.data'
    mldata4 = loadheartdisease(path4)
    sampledmldata4 = mldata4.sampledata({'train':70, 'test':10, 'cv':20})
    dt4 = dtree(sampledmldata4['train'])
    dt4.generatetree(dt4.root)
    print "before pruning...."
    dt4.printtree()
    dt4.testtree(sampledmldata4['test'], False, True)
    #print "after pruning...."
    #besttree4 = dt4.prune(2, sampledmldata4['cv'], True)
    #besttree4.testtree(sampledmldata4['test'], False, True)
    #dt4.prune(1, sampledmldata4['cv'], True)
    #dt4.testtree(sampledmldata4['test'], False, True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
