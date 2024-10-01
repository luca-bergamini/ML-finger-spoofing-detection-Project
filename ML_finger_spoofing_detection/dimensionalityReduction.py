#PCA determina, partendo da un vettore n-dimensionale, un vettore a 2 dimensioni
#calcolando n PC (PC1, PC2, ..) a seconda di quante dimensioni ci sono, trovando poi
#la rispettiva variazione che i diversi PCn portano ai dati, infatti un dato PCx varierà
#l'intero insieme dei dati di più ripetto ad un altro PCy. Vengono presi i due PCn che
#variano di più i dati e tramite questi due viene creato il vettore a 2 dimensioni, usando
#i 2 PCn come x e y di un grafico.
#per varierà si intende che è più responsabile rispetto ad un altro PCx di variare i dati
import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.datasets

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

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def makeHist(D, L, x, bins=100):
    #maschere per dividere D secondo la label L
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    BINS = bins
    plt.figure() #per creare un nuovo grafico
    plt.hist(D0[x, :], bins=BINS, density=True, alpha=0.4, label="FALSE")
    plt.hist(D1[x, :], bins=BINS, density=True, alpha=0.4, label="TRUE")
    #plt.xlabel("Direction " + str(x+1))
    plt.legend()
    plt.show()
    
def makeScatter(D, L, x1, x2):
    #maschere per dividere D secondo la label L
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    plt.figure() #per creare un nuovo grafico
    plt.scatter(D0[x1, :], D0[x2, :], label="FALSE", alpha=0.4)
    plt.scatter(D1[x1, :], D1[x2, :], label="TRUE", alpha=0.4)
    plt.xlabel("Direction " + str(x1+1))
    plt.ylabel("Direction " + str(x2+1))
    plt.legend()
    plt.show()
    
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
        
#calcolo dei pesi delle dimensioni, W è un array di pesi
def LDA(D, L, m):
    SB, SW = covarianceMatrix(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m] #W sono i pesi delle dimensioni
    #W non è necessariamente ortogonale, per renderla tale si possono usare le due righe sottostanti
    #UW, _, _ = numpy.linalg.svd(W)
    #U = UW[:, 0:m]
    return W

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
        
def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    
#metodo per dividere il database in 2 il database, una parte di training con 2/3 dei dati e una parte di validazione dei dati con 1/3 dei dati
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def predict_labels(DTR, LTR, DVAL, LVAL):
    W = LDA(DTR, LTR, 6)
    DTR_lda = apply_LDA(W, DTR)
    DVAL_lda = apply_LDA(W, DVAL)
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 #-0.084
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    #da pdf il valore TRUE è più grande, quindi tutto ciò che c'è a destra del threshold è TRUE cioè 1
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    print("Threshold: ", threshold)
    return PVAL

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

if __name__ == '__main__':
    D, L = load("trainData.txt")
    #P = PCA(D, 5)
    #DP = apply_PCA(P, D)
    #makeHist(DP, L, 5)
    #makeScatter(DP, L, 4, 5)
    #W = LDA(D,L,6)
    #DLDA = apply_LDA(W, D)
    #makeHist(DLDA, L, 0)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    #W = LDA(DTR, LTR, 6)
    #DLDA = apply_LDA(W, DVAL)
    #P = predict_labels(DTR, LTR, DVAL, LVAL)
    #er, suc, er_rate, succ_rate = calculate_rates(P, LVAL)
    #print("errors: ", er)    
    #print("success: ", suc)
    #print("er_rate: ", er_rate)
    #print("succ_rate: ", succ_rate)
    UPCA = PCA(DTR, 2)
    DTR_pca = apply_PCA(UPCA, DTR)
    DVAL_pca = apply_PCA(UPCA, DVAL)
    
    ULDA = LDA_diag(DTR_pca, LTR, 1)
    DTR_lda = apply_LDA(ULDA, DTR_pca)
    if DTR_lda[0, LTR==0].mean() > DTR_lda[0, LTR==1].mean():
        ULDA = -ULDA
        DTR_lda = apply_LDA(ULDA, DTR_pca)
        
    DVAL_lda = apply_LDA(ULDA, DVAL_pca)
    
    threshold = 0.02#(DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
    