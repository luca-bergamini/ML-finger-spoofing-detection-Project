import numpy
import scipy.special

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

#calcolo le matrici di covarianza
def covarianceMatrix(D, L):
    mu = D.mean(1)
    n_feature, n_sample = D.shape #prendo il numero di righe e colonne per inizializzare il vettore risultante
    unique = numpy.unique(L) #array di interi con i diversi valori che ci sono in L, quindi nel caso del dataset prendendo solo le classi 1 e 2 ho unique che è [1, 2]
                             #quindi i un ciclo sarà 1 e l'altro sarà 2
    SW = SB = numpy.zeros((n_feature, n_feature))
    for i in unique:
        DLab = D[:, L==i]
        muDLab = DLab.mean(1)
        diff_mu = vcol(muDLab) - vcol(mu) #per calcolare SB devo fare la sommatoria di (mu della classe - mu generale)
        DC = DLab - vcol(muDLab)
        nc = DLab.shape[1]
        SB = SB + (nc * (diff_mu@diff_mu.T))
        SW = SW + (DC@DC.T)
    SB = SB / float(n_sample)
    SW = SW / float(n_sample)
    return SB, SW

def LDA_diag(D, L, m):
    SB, SW = covarianceMatrix(D, L)
    U, s, _ = numpy.linalg.svd(SW)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(SB, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

#la funzione apply_LDA la faccio separata perchè la LDA la voglio calcolare usando i pesi calcolati
#con il dataset di training, sul dataset di valutazione DVAL
def apply_LDA(W, D):
    return W.T @ D #per applicare i pesi su tutta la matrice

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

def calculate_rates(PVAL, LVAL):
    error = 0
    success = 0
    for i in range(len(PVAL)):
        if PVAL[i] != LVAL[i]:
            error += 1
        else:
            success += 1
    tot = error + success
    return error, success, error/tot, success/tot
    
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


if __name__ == '__main__':
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    TH = 0 #uguale a zero perche si calcola come -log(prob per la classe 1/ prob per la classe 0), e tutte e due hanno prob 1/2 quindi viene 0
    
    #-------------- MVG
    params_MVG = gaussian_MVG_estimations(DTR, LTR)
    llr_MVG = logpdf_GAU_ND(DVAL, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL, params_MVG[0][0], params_MVG[0][1])
    PVAL_MVG = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_MVG[llr_MVG >= TH] = 1
    PVAL_MVG[llr_MVG < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_MVG, LVAL)
    print("MVG")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("MVG - Error rate: %.1f%%" % ((PVAL_MVG != LVAL).sum() / float(LVAL.size) * 100))
    print()
 
    #-------------- LDA
    UPCA = PCA(DTR, 6)
    DTR_pca = apply_PCA(UPCA, DTR)
    DVAL_pca = apply_PCA(UPCA, DVAL)
    
    ULDA = LDA_diag(DTR_pca, LTR, 1)
    DTR_lda = apply_LDA(ULDA, DTR_pca)
    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = apply_LDA(ULDA, DTR_pca)
        
    DVAL_lda = apply_LDA(ULDA, DVAL_pca)
    PVAL_LDA = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_LDA[DVAL_lda[0] >= TH] = 1
    PVAL_LDA[DVAL_lda[0] < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_LDA, LVAL)
    print("LDA")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("LDA - Error rate: %.1f%%" % ((PVAL_LDA != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- TIED
    params_tied = gaussian_Tied_estimations(DTR, LTR)
    #params_tied[x][y] -> x = label, y = 0 -> mu oppure 1 -> C
    llr_tied = logpdf_GAU_ND(DVAL, params_tied[1][0], params_tied[1][1]) - logpdf_GAU_ND(DVAL, params_tied[0][0], params_tied[0][1])
    PVAL_TIED = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_TIED[llr_tied >= TH] = 1
    PVAL_TIED[llr_tied < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_TIED, LVAL)
    print("TIED")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Tied - Error rate: %.1f%%" % ((PVAL_TIED != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- NAIVE
    params_naive = gaussian_Naive_estimations(DTR, LTR)
    llr_naive = logpdf_GAU_ND(DVAL, params_naive[1][0], params_naive[1][1]) - logpdf_GAU_ND(DVAL, params_naive[0][0], params_naive[0][1])
    PVAL_NAIVE = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_NAIVE[llr_naive >= TH] = 1
    PVAL_NAIVE[llr_naive < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_NAIVE, LVAL)
    print("NAIVE")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Naive - Error rate: %.1f%%" % ((PVAL_NAIVE != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- Covariance matrix
    C_0 = params_MVG[0][1]
    C_1 = params_MVG[1][1]
    Corr_0 = C_0 / (vcol(C_0.diagonal()**0.5) * vrow(C_0.diagonal()**0.5))
    Corr_1 = C_1 / (vcol(C_1.diagonal()**0.5) * vrow(C_1.diagonal()**0.5))
    numpy.set_printoptions(precision=2, suppress=True)
    # print("Correlation Matrix 0: ", Corr_0)
    # print("Correlation Matrix 1: ", Corr_1)    
    
    #-------------- Classification feature 1 to 4
    D04 = D[0:4]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D04, L)
    TH = 0
    
    #-------------- MVG
    params_MVG = gaussian_MVG_estimations(DTR, LTR)
    llr_MVG = logpdf_GAU_ND(DVAL, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL, params_MVG[0][0], params_MVG[0][1])
    PVAL_MVG = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_MVG[llr_MVG >= TH] = 1
    PVAL_MVG[llr_MVG < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_MVG, LVAL)
    print("MVG")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("MVG - Error rate: %.1f%%" % ((PVAL_MVG != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- LDA
    UPCA = PCA(DTR, 4)
    DTR_pca = apply_PCA(UPCA, DTR)
    DVAL_pca = apply_PCA(UPCA, DVAL)
    
    ULDA = LDA_diag(DTR_pca, LTR, 1)
    DTR_lda = apply_LDA(ULDA, DTR_pca)
    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = apply_LDA(ULDA, DTR_pca)
        
    DVAL_lda = apply_LDA(ULDA, DVAL_pca)
    PVAL_LDA = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_LDA[DVAL_lda[0] >= TH] = 1
    PVAL_LDA[DVAL_lda[0] < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_LDA, LVAL)
    print("LDA")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("LDA - Error rate: %.1f%%" % ((PVAL_LDA != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- TIED
    params_tied = gaussian_Tied_estimations(DTR, LTR)
    #params_tied[x][y] -> x = label, y = 0 -> mu oppure 1 -> C
    llr_tied = logpdf_GAU_ND(DVAL, params_tied[1][0], params_tied[1][1]) - logpdf_GAU_ND(DVAL, params_tied[0][0], params_tied[0][1])
    PVAL_TIED = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_TIED[llr_tied >= TH] = 1
    PVAL_TIED[llr_tied < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_TIED, LVAL)
    print("TIED")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Tied - Error rate: %.1f%%" % ((PVAL_TIED != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- NAIVE
    params_naive = gaussian_Naive_estimations(DTR, LTR)
    llr_naive = logpdf_GAU_ND(DVAL, params_naive[1][0], params_naive[1][1]) - logpdf_GAU_ND(DVAL, params_naive[0][0], params_naive[0][1])
    PVAL_NAIVE = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_NAIVE[llr_naive >= TH] = 1
    PVAL_NAIVE[llr_naive < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_NAIVE, LVAL)
    print("NAIVE")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Naive - Error rate: %.1f%%" % ((PVAL_NAIVE != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- Classification feature 1 and 2
    print("----- Classification feature 1 and 2")
    D01 = D[0:2]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D01, L)
    D0F0 = D[0, L==0]
    D0F1 = D[0, L==1]
    D1F0 = D[1, L==0]
    D1F1 = D[1, L==1]
    TH = 0
    
    print("Variance 1 true: ", D0F0.var())
    print("Variance 1 false: ", D0F1.var())
    print("Variance 2 true: ", D1F0.var())
    print("Variance 2 false: ", D1F1.var())
    print()
    
    #-------------- MVG
    params_MVG = gaussian_MVG_estimations(DTR, LTR)
    llr_MVG = logpdf_GAU_ND(DVAL, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL, params_MVG[0][0], params_MVG[0][1])
    PVAL_MVG = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_MVG[llr_MVG >= TH] = 1
    PVAL_MVG[llr_MVG < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_MVG, LVAL)
    print("MVG")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("MVG - Error rate: %.1f%%" % ((PVAL_MVG != LVAL).sum() / float(LVAL.size) * 100))
    print("var 0: ", D[:, L==0].var())
    print("var 1: ", params_MVG[1][1].var())
    print()
    
    #-------------- TIED
    params_tied = gaussian_Tied_estimations(DTR, LTR)
    #params_tied[x][y] -> x = label, y = 0 -> mu oppure 1 -> C
    llr_tied = logpdf_GAU_ND(DVAL, params_tied[1][0], params_tied[1][1]) - logpdf_GAU_ND(DVAL, params_tied[0][0], params_tied[0][1])
    PVAL_TIED = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_TIED[llr_tied >= TH] = 1
    PVAL_TIED[llr_tied < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_TIED, LVAL)
    print("TIED")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Tied - Error rate: %.1f%%" % ((PVAL_TIED != LVAL).sum() / float(LVAL.size) * 100))
    print("var 0: ", params_tied[0][1].var())
    print("var 1: ", params_tied[1][1].var())
    print()
    
    #-------------- Classification feature 3 and 4
    print("----- Classification feature 3 and 4")
    D34 = D[2:4]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D34, L)
    D2F0 = D[2, L==0]
    D2F1 = D[2, L==1]
    D3F0 = D[3, L==0]
    D3F1 = D[3, L==1]
    TH = 0
    
    print("Variance 3 true: ", D2F0.var())
    print("Variance 3 false: ", D2F1.var())
    print("Variance 4 true: ", D3F0.var())
    print("Variance 4 false: ", D3F1.var())
    print()
    
    #-------------- MVG
    params_MVG = gaussian_MVG_estimations(DTR, LTR)
    llr_MVG = logpdf_GAU_ND(DVAL, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL, params_MVG[0][0], params_MVG[0][1])
    PVAL_MVG = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_MVG[llr_MVG >= TH] = 1
    PVAL_MVG[llr_MVG < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_MVG, LVAL)
    print("MVG")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("MVG - Error rate: %.1f%%" % ((PVAL_MVG != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- TIED
    params_tied = gaussian_Tied_estimations(DTR, LTR)
    #params_tied[x][y] -> x = label, y = 0 -> mu oppure 1 -> C
    llr_tied = logpdf_GAU_ND(DVAL, params_tied[1][0], params_tied[1][1]) - logpdf_GAU_ND(DVAL, params_tied[0][0], params_tied[0][1])
    PVAL_TIED = numpy.zeros(DVAL.shape[1], dtype=numpy.int32)
    PVAL_TIED[llr_tied >= TH] = 1
    PVAL_TIED[llr_tied < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_TIED, LVAL)
    print("TIED")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Tied - Error rate: %.1f%%" % ((PVAL_TIED != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- PCA for pre-processing -> MVG, Tied e Naive
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    TH = 0
    
    m = 6
    UPCA = PCA(DTR, m)
    DTR_pca = apply_PCA(UPCA, DTR)
    DVAL_pca = apply_PCA(UPCA, DVAL)
    
    print("Models after usering PCA with m = ", m)
    print()
    
    #-------------- MVG
    params_MVG = gaussian_MVG_estimations(DTR_pca, LTR)
    llr_MVG = logpdf_GAU_ND(DVAL_pca, params_MVG[1][0], params_MVG[1][1]) - logpdf_GAU_ND(DVAL_pca, params_MVG[0][0], params_MVG[0][1])
    PVAL_MVG = numpy.zeros(DVAL_pca.shape[1], dtype=numpy.int32)
    PVAL_MVG[llr_MVG >= TH] = 1
    PVAL_MVG[llr_MVG < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_MVG, LVAL)
    print("MVG")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("MVG - Error rate: %.1f%%" % ((PVAL_MVG != LVAL).sum() / float(LVAL.size) * 100))
    print()

    #-------------- TIED
    params_tied = gaussian_Tied_estimations(DTR_pca, LTR)
    #params_tied[x][y] -> x = label, y = 0 -> mu oppure 1 -> C
    llr_tied = logpdf_GAU_ND(DVAL_pca, params_tied[1][0], params_tied[1][1]) - logpdf_GAU_ND(DVAL_pca, params_tied[0][0], params_tied[0][1])
    PVAL_TIED = numpy.zeros(DVAL_pca.shape[1], dtype=numpy.int32)
    PVAL_TIED[llr_tied >= TH] = 1
    PVAL_TIED[llr_tied < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_TIED, LVAL)
    print("TIED")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Tied - Error rate: %.1f%%" % ((PVAL_TIED != LVAL).sum() / float(LVAL.size) * 100))
    print()
    
    #-------------- NAIVE
    params_naive = gaussian_Naive_estimations(DTR_pca, LTR)
    llr_naive = logpdf_GAU_ND(DVAL_pca, params_naive[1][0], params_naive[1][1]) - logpdf_GAU_ND(DVAL_pca, params_naive[0][0], params_naive[0][1])
    PVAL_NAIVE = numpy.zeros(DVAL_pca.shape[1], dtype=numpy.int32)
    PVAL_NAIVE[llr_naive >= TH] = 1
    PVAL_NAIVE[llr_naive < TH] = 0
    er, suc, er_rate, succ_rate = calculate_rates(PVAL_NAIVE, LVAL)
    print("NAIVE")
    print("errors: ", er)    
    print("success: ", suc)
    print("er_rate: ", er_rate)
    print("succ_rate: ", succ_rate)
    print("Naive - Error rate: %.1f%%" % ((PVAL_NAIVE != LVAL).sum() / float(LVAL.size) * 100))
    print()









