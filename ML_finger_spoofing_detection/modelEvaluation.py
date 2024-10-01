import numpy
from sklearn.metrics import confusion_matrix
import scipy.special
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

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)
    #DTR e LTR sono i dati di training con le loro label
    #DTE e LTE sono i dati di validazione con le loro label
    
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def PCA(D, m):
    mu = D.mean(1)
    #DC = matrix of centered data
    DC = D - vcol(mu) #per togliere il valore medio da tutte le colonne
    C = (DC @ DC.T) / float(D.shape[1]) #matrice di covarianza e indica quanto due variabili sono legate
                                        #>0 quando uno cresce la'tro cresce, <0viceversa (una aumenta l'altra diminuisce), =0 non ce relazione
    #print("mu: ", mu)
    #print("C: ", C)
    
    #Gli autovalori della matrice di covarianza indicano la variabilità o l'importanza di ogni direzione dei dati
    #autovalore grande indica molta variazione dei dati, autovalore piccolo indica poca variazione
    #Gli autovettori indicano le direzioni nello spazio dei dati che catturano la maggior parte delle variazioni dei dati
    #numpy.linalg.eig e numpy.linalg.eigh calcolano autovalori e autovettori ma numpy.linalg.eigh è più precisa quando 
    #si lavora con matrici simmetriche e numpy.linalg.eig non ordina i valori
    #------------- PRIMO METODO PER CALCOLARE s, U, P
    #s = array di autovalori, in ordine crescente
    #U = matrice di autovettori
    s, U = numpy.linalg.eigh(C)
    #vogliamo prendere i primi m autovettori, invertiamo l'ordine delle colonne di U (U[:, ::-1]), non le righe
    #poi per ogni riga prendo da 0 a m colonne ([:, 0:m]), il per ogni riga deriva dal primo :
    P = U[:, ::-1][:, 0:m]
    #------------- SECONDO METODO PER calcolare autovettori usando SVD (Singular Value Decomposition)
    #U, s, Vh = numpy.linalg.svd(C) #valori sortati in ordine decrescente
    #P = U[:, 0:m]
    #possiamo ora applicare la proiezione o su un punto x
    #y = numpy.dot(P.T, x)
    #oppure su una matrice di samples
    return P

def apply_PCA(P,D):
    return P.T @ D
    
def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

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

def gaussian_MVG_estimations(D, L):
    labels = set(L) #mi tira fuori le diverse occorrenze presenti in L
    params = {}
    for label in labels:
        Dl = D[:, L==label]
        params[label] = compute_mu_C(Dl) #params per ogni label contiene una tupla (mu, C)
    return params

def compute_log_likelihoods(D, params): #params = valore di ritorno della gaussian_MVG_estinations
    S = numpy.zeros((len(params), D.shape[1]))
    for label in range(S.shape[0]): #metto 0 perchè voglio il range per il numero di classi quindi il primo indice di S
        S[label, :] = logpdf_GAU_ND(D, params[label][0], params[label][1]) #0 contiene la mu, 1 contiene la C
    return S     
    
def compute_logPosterior(S, prob):
    logSJoint = S + vcol(numpy.log(prob)) #facciamo questo perchè abbiamo calcolato il log del likelihood
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    #SPost = numpy.exp(logSPost) questo lo faccio se voglio SPost e non il log
    return logSPost

def gaussian_Naive_estimations(D, L):
    labels = set(L)
    params = {}
    for label in labels:
        Dl = D[:, L==label]
        mu, C = compute_mu_C(Dl)
        params[label] = (mu, C * numpy.eye(D.shape[0])) #Il risultato sarà una nuova matrice con gli stessi valori di C, ma con 1 sulla diagonale e 0 altrove
    return params
    
def gaussian_Tied_estimations(D, L):
    labels = set(L)
    params = {}
    means = {}
    Ctot = 0
    for label in labels:
        Dl = D[:, L==label]
        mu, C = compute_mu_C(Dl)
        Ctot += C * Dl.shape[1]
        means[label] = mu
    Ctot = Ctot / D.shape[1]
    for label in labels:
        params[label] = (means[label], Ctot)
    return params

def costPredictor(p, Cfn, Cfp):
    threshold = -numpy.log((p*Cfn)/((1-p)*Cfp))
    return threshold
    
def binaryDCF(p, Cfn, Cfp, confusion_matrix): #detection cost function -> costo che dobbiamo pagare per le nostre decisioni (c) per i test data (la funzione serve per comparare diversi sistemi)
    Pfn = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]) #false negative rate -> FN / FN + TP -> FN = confusion_matrix[0][1] 
    Pfp = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0]) #false positive rate -> FP / FP + TN
    DFC = p * Cfn * Pfn + (1-p) * Cfp * Pfp
    return DFC

def binaryNormalizedDCF(p, Cfn, Cfp, confusion_matrix): #serve per comparare diversi sistemi, ci dice anche qual è il beneficio di usare un certo riconizer rispetto alle decisioni ottimali
    Pfn = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]) #false negative rate -> FN / FN + TP -> FN = confusion_matrix[0][1] 
    Pfp = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0]) #false positive rate -> FP / FP + TN
    DFC = p * Cfn * Pfn + (1-p) * Cfp * Pfp
    DFCDummy = min(p*Cfn, (1-p)*Cfp)
    return DFC/DFCDummy

def minDCF(llr, thresholds, labels, p, Cfn, Cfp): #prendiamo il minor DFC calcolato con ogni threshold, il minor DFC sarebbe il migliore possibile
    DCFarr = []
    thresholds = sorted(thresholds, reverse=True)
    for threshold in thresholds:
        PVAL = numpy.where(llr > threshold, 1, 0)
        conf_mat = confusion_matrix(labels, PVAL, labels=[0,1]).T
        DCFarr.append(binaryNormalizedDCF(p, Cfn, Cfp, conf_mat))
    return min(DCFarr)

def ROC(llr, thresholds, labels):
    PfpA = []
    PtpA = []
    thresholds = sorted(thresholds, reverse=True)
    for threshold in thresholds:
        PVAL = numpy.where(llr > threshold, 1, 0)
        conf_mat = confusion_matrix(labels, PVAL, labels=[0,1]).T
        Pfn = conf_mat[0][1] / (conf_mat[0][1] + conf_mat[1][1])
        Pfp = conf_mat[1][0] / (conf_mat[0][0] + conf_mat[1][0])
        Ptp = 1 - Pfn
        PfpA.append(Pfp)
        PtpA.append(Ptp)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(PfpA, PtpA)
    plt.grid(True)
    plt.show()
    
def bayesError(llr, thresholds, pi, labels, eff):
    dcf = []
    mindcf = []
    for threshold in pi:
        PVAL = numpy.where(llr > threshold, 1, 0)
        conf_mat = confusion_matrix(labels, PVAL, labels=[0,1]).T
        DCF = binaryNormalizedDCF(threshold, 1, 1, conf_mat)
        MINDCF = minDCF(llr, thresholds, labels, threshold, 1, 1)
        dcf.append(DCF)
        mindcf.append(MINDCF)
    
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.plot(eff, dcf, label='DCF', color='r')
    plt.plot(eff, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend(loc='lower left')
    plt.show()
    

if __name__ == '__main__':
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    #-------------- MVG without PCA
    params_MVG = gaussian_MVG_estimations(DTR, LTR)

    #-------------- Tied without PCA
    params_tied = gaussian_Tied_estimations(DTR, LTR)
    
    #-------------- Naive without PCA
    params_naive = gaussian_Naive_estimations(DTR, LTR)
    
    '''
    #prima parte del progetto, soluzione per i 5 punti, cambiando i parametri sottostanti
    p = [0.5, 0.9, 0.1, 0.5, 0.5]
    Cfn = [1, 1, 1, 1.0, 9.0]
    Cfp = [1, 1, 1, 9.0, 1.0]
    
    for i in range(0, 5):
        TH = costPredictor(p[i], Cfn[i], Cfp[i])
        print("threshold: ", i, " -> ", TH)
        llr_MVG = logpdf_GAU_ND(DVAL, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL, params_MVG[0][0], params_MVG[0][1])
        PVAL_C = numpy.where(llr_MVG > TH, 1, 0)
        conf_mat_C = confusion_matrix(LVAL, PVAL_C, labels=[0,1]).T
        print("Confusion matrix Cost of ", i)
        print(conf_mat_C)
        DCF = binaryDCF(p[i], Cfn[i], Cfp[i], conf_mat_C)
        print("DCF: ", DCF)
        normalizedDCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat_C)
        print("Normalized DCF: ", normalizedDCF)
        mindcf = minDCF(llr_MVG, llr_MVG, LVAL, p[i], Cfn[i], Cfp[i])
        print("Min DCF: ", mindcf)
    '''
    
    p = [0.5, 0.9, 0.1, 0.5, 0.5]
    Cfn = [1, 1, 1, 1.0, 9.0]
    Cfp = [1, 1, 1, 9.0, 1.0]
    
    i = 2
    
    '''
    #-------------- MVG
    llr_MVG = logpdf_GAU_ND(DVAL, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL, params_MVG[0][0], params_MVG[0][1])
    print("Config: ", p[i], Cfn[i], Cfp[i])
    th = costPredictor(p[i], Cfn[i], Cfp[i])
    PVAL = numpy.where(llr_MVG > th, 1, 0)
    conf_mat = confusion_matrix(LVAL, PVAL).T
    DCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat)
    DCFmin = minDCF(llr_MVG, llr_MVG, LVAL, p[i], Cfn[i], Cfp[i])
    print("MVG: ", end=' ')
    print(conf_mat, DCF, DCFmin)
    print("Miscalibration percentage: ", (1 - DCFmin / DCF) * 100)
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    #bayesError(llr_MVG, llr_MVG, pi, LVAL, effPriorLogOdds)
    
    #-------------- Tied
    llr_tied = logpdf_GAU_ND(DVAL, params_tied[1][0], params_tied[1][1]) - logpdf_GAU_ND(DVAL, params_tied[0][0], params_tied[0][1])
    print("Config: ", p[i], Cfn[i], Cfp[i])
    th = costPredictor(p[i], Cfn[i], Cfp[i])
    PVAL = numpy.where(llr_tied > th, 1, 0)
    conf_mat = confusion_matrix(LVAL, PVAL).T
    DCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat)
    DCFmin = minDCF(llr_tied, llr_tied, LVAL, p[i], Cfn[i], Cfp[i])
    print("TIED: ", end=' ')
    print(conf_mat, DCF, DCFmin)
    print("Miscalibration percentage: ", (1 - DCFmin / DCF) * 100)
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    #bayesError(llr_tied, llr_tied, pi, LVAL, effPriorLogOdds)
    
    #-------------- Naive
    llr_naive = logpdf_GAU_ND(DVAL, params_naive[1][0], params_naive[1][1]) - logpdf_GAU_ND(DVAL, params_naive[0][0], params_naive[0][1])
    print("Config: ", p[i], Cfn[i], Cfp[i])
    th = costPredictor(p[i], Cfn[i], Cfp[i])
    PVAL = numpy.where(llr_naive > th, 1, 0)
    conf_mat = confusion_matrix(LVAL, PVAL).T
    DCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat)
    DCFmin = minDCF(llr_naive, llr_naive, LVAL, p[i], Cfn[i], Cfp[i])
    print("Naive: ", end=' ')
    print(conf_mat, DCF, DCFmin)
    print("Miscalibration percentage: ", (1 - DCFmin / DCF) * 100)
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    #bayesError(llr_naive, llr_naive, pi, LVAL, effPriorLogOdds)
    '''
    
    
    #-------------- Calcolo PCA
    m = 1
    UPCA = PCA(DTR, m)
    DTR_pca = apply_PCA(UPCA, DTR)
    DVAL_pca = apply_PCA(UPCA, DVAL)
    
    #-------------- MVG with PCA
    params_MVG_pca = gaussian_MVG_estimations(DTR_pca, LTR)

    #-------------- Tied with PCA
    params_tied_pca = gaussian_Tied_estimations(DTR_pca, LTR)
    
    #-------------- Naive with PCA
    params_naive_pca = gaussian_Naive_estimations(DTR_pca, LTR)
    
    #-------------- MVG PCA
    llr_MVG = logpdf_GAU_ND(DVAL_pca, params_MVG_pca[1][0], params_MVG_pca[1][1]) - logpdf_GAU_ND(DVAL_pca, params_MVG_pca[0][0], params_MVG_pca[0][1])
    print("Config: ", p[i], Cfn[i], Cfp[i])
    th = costPredictor(p[i], Cfn[i], Cfp[i])
    PVAL = numpy.where(llr_MVG > th, 1, 0)
    conf_mat = confusion_matrix(LVAL, PVAL, labels=[0,1]).T
    DCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat)
    DCFmin = minDCF(llr_MVG, llr_MVG, LVAL, p[i], Cfn[i], Cfp[i])
    print("MVG: ", end=' ')
    print(conf_mat, "DCF: ", DCF, "DCFmin: ", DCFmin)
    print("Miscalibration percentage: ", (1 - DCFmin / DCF) * 100)
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    #bayesError(llr_MVG, llr_MVG, pi, LVAL, effPriorLogOdds)
    
    #-------------- Tied PCA
    llr_tied = logpdf_GAU_ND(DVAL_pca, params_tied_pca[1][0], params_tied_pca[1][1]) - logpdf_GAU_ND(DVAL_pca, params_tied_pca[0][0], params_tied_pca[0][1])
    print("Config: ", p[i], Cfn[i], Cfp[i])
    th = costPredictor(p[i], Cfn[i], Cfp[i])
    PVAL = numpy.where(llr_tied > th, 1, 0)
    conf_mat = confusion_matrix(LVAL, PVAL, labels=[0,1]).T
    DCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat)
    DCFmin = minDCF(llr_tied, llr_tied, LVAL, p[i], Cfn[i], Cfp[i])
    print("TIED: ", end=' ')
    print(conf_mat, "DCF: ", DCF, "DCFmin: ", DCFmin)
    print("Miscalibration percentage: ", (1 - DCFmin / DCF) * 100)
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    #bayesError(llr_tied, llr_tied, pi, LVAL, effPriorLogOdds)
    
    #-------------- Naive PCA
    llr_naive = logpdf_GAU_ND(DVAL_pca, params_naive_pca[1][0], params_naive_pca[1][1]) - logpdf_GAU_ND(DVAL_pca, params_naive_pca[0][0], params_naive_pca[0][1])
    print("Config: ", p[i], Cfn[i], Cfp[i])
    th = costPredictor(p[i], Cfn[i], Cfp[i])
    PVAL = numpy.where(llr_naive > th, 1, 0)
    conf_mat = confusion_matrix(LVAL, PVAL, labels=[0,1]).T
    DCF = binaryNormalizedDCF(p[i], Cfn[i], Cfp[i], conf_mat)
    DCFmin = minDCF(llr_naive, llr_naive, LVAL, p[i], Cfn[i], Cfp[i])
    print("Naive: ", end=' ')
    print(conf_mat, "DCF: ", DCF, "DCFmin: ", DCFmin)
    print("Miscalibration percentage: ", (1 - DCFmin / DCF) * 100)
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    #bayesError(llr_naive, llr_naive, pi, LVAL, effPriorLogOdds)
    
    
    
    
    
    
    
    
    
    
    