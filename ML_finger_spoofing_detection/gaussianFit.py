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

def vrow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))

def logpdf_GAU_ND_x(x, mu, C):
    M = C.shape[0]
    i, logC = numpy.linalg.slogdet(C)
    N = -(M/2)*numpy.log(2*numpy.pi) - (1/2)*logC - (1/2)*((x-mu).T@numpy.linalg.inv(C)@(x-mu))
    return N

def logpdf_GAU_ND(X, mu, C):
    N = X.shape[1]
    Y = numpy.zeros(N)
    for i in range(N):
        Y[i] = logpdf_GAU_ND_x(X[:, i:i+1], mu, C) #metto X[:, i] perche prendo tutte le righe ma solo la i colonna
    return numpy.array(Y).ravel()

def mu_C_ml(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def loglikelihood(X, mu, C):
    res = 0
    for x in logpdf_GAU_ND(X, mu, C):
        res += x
    return res

if __name__ == '__main__':
    D, L = load('trainData.txt')  # Carica i dati dal file
    print(D.shape)  # Stampa la forma dei dati
    muTRUE, CTRUE = mu_C_ml(D[:, L == 1])  # Calcola media e covarianza
    muFALSE, CFALSE = mu_C_ml(D[:, L == 0])  # Calcola media e covarianza
    res = numpy.zeros((6, 1))
    plt.figure()  # Inizializza una nuova figura

    # Loop attraverso le feature per plottare l'istogramma e la distribuzione Gaussiana
    for i in range(D.shape[0]):
        plt.figure()
        plt.title("Feature " + str(i+1))        
        XPlot = numpy.linspace(-4.5, 4.5, 250)  # Genera punti per il plot della distribuzione
        plt.hist(D[i,L == 1], bins=100, alpha=0.4, density=True, label="TRUE")  # Istogramma della feature i con L == 0
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), muTRUE[i], CTRUE[i:i+1, i:i+1])), label="TRUE")  # Plot della distribuzione Gaussiana
        plt.hist(D[i,L == 0], bins=100, alpha=0.4, density=True, label="FALSE")  # Istogramma della feature i con L == 0
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), muFALSE[i], CFALSE[i:i+1, i:i+1])), label = "FALSE")  # Plot della distribuzione Gaussiana
        plt.legend()
        
    
    
    
    
    
    