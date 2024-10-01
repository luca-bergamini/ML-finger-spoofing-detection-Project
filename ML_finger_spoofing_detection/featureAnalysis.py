import numpy
import matplotlib.pyplot as plt

def load(fname):
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            attrs = line.split(",")[0:-1]
            attrs = numpy.array(attrs, dtype=numpy.float32) #crea un numpy array con elementi float32
            attrs = attrs.reshape((attrs.size, 1)) #fa il reshape creando una colonna con n righe
            name = line.split(",")[-1].strip(); #.strip elimina gli spazi e gli a capo
            DList.append(attrs)
            labelsList.append(name)
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def makeHist(D, L, x, bins=100):
    #maschere per dividere D secondo la label L
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    BINS = bins
    plt.figure() #per creare un nuovo grafico
    plt.hist(D0[x, :], bins=BINS, density=True, alpha=0.4, label="FALSE")
    plt.hist(D1[x, :], bins=BINS, density=True, alpha=0.4, label="TRUE")
    plt.xlabel("Feature " + str(x))
    plt.legend()
    plt.show()
    
def makeScatter(D, L, x1, x2):
    #maschere per dividere D secondo la label L
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    plt.figure() #per creare un nuovo grafico
    plt.scatter(D0[x1, :], D0[x2, :], label="FALSE")
    plt.scatter(D1[x1, :], D1[x2, :], label="TRUE")
    plt.xlabel("Feature " + str(x1))
    plt.ylabel("Feature " + str(x2))
    plt.legend()
    plt.show()
    
def datasetMeans(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu #per togliere il valore medio dai punti
    return DC

def meanVarianceSubset(D, L, feature):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    feature0 = D0[feature, :] #prende l'array relativo alla feature indicata
    feature1 = D1[feature, :]
    print("Mean for feature %s with label FALSE: %s" %(feature, feature0.mean()))
    print("Mean for feature %s with label TRUE: %s" %(feature, feature1.mean()))
    print("Variance for feature %s with label FALSE: %s" %(feature, feature0.var()))
    print("Variance for feature %s with label TRUE: %s" %(feature, feature1.var()))
    return feature0.mean(), feature1.mean(), feature0.var(), feature1.var()
    
    
if __name__ == '__main__':
    D, L = load("trainData.txt")
    makeHist(D, L, 5)
    #makeScatter(D, L, 4, 5)
    meanVarianceSubset(D, L, 4)
    meanVarianceSubset(D, L, 5)