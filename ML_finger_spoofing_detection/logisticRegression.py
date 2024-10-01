"""
Created on Wed Aug 14 11:31:01 2024

@author: LucaBergamini
"""

import numpy
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
    
def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def whiten_data(DTR, DTE):
    mu, C = compute_mu_C(DTR)
    U, s, _ = numpy.linalg.svd(C)
    W = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)
    DTR_whitened = W @ DTR
    DTE_whitened = W @ DTE
    return DTR_whitened, DTE_whitened

def expand_quadratic_features(DTR):
    num_features = DTR.shape[0]
    expanded_features = []
    # Original features
    expanded_features.append(DTR)
    # Quadratic terms
    for i in range(num_features):
        for j in range(num_features): #i, num_features
            expanded_features.append(DTR[i, :] * DTR[j, :])

    return numpy.vstack(expanded_features)

def center_data(DTR, DTE):
    mu = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mu
    DTE_centered = DTE - mu
    return DTR_centered, DTE_centered

def z_normalize_data(DTR, DTE):
    mu = DTR.mean(axis=1, keepdims=True)
    sigma = DTR.std(axis=1, keepdims=True)
    DTR_normalized = (DTR - mu) / sigma
    DTE_normalized = (DTE - mu) / sigma
    return DTR_normalized, DTE_normalized

def PCA(D, m):
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = (DC @ DC.T) / float(D.shape[1]) 
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P

def apply_PCA(DTR, DTE, m):
    P = PCA(DTR, m)
    DTR_pca = P.T @ DTR
    DTE_pca = P.T @ DTE
    return DTR_pca, DTE_pca

def cost_predictor(p, Cfn, Cfp):
    threshold = -numpy.log((p*Cfn)/((1-p)*Cfp))
    return threshold

def compute_confusion_matrix(predicted_labels, LVAL):
    unique_classes = len(numpy.unique(LVAL))
    confusion_matrix = numpy.zeros((unique_classes, unique_classes))
    for i in range(len(predicted_labels)):
        confusion_matrix[int(predicted_labels[i])][int(LVAL[i])] += 1
    return confusion_matrix

def compute_binary_normalized_DCF(predictedLabels, LVAL, p, Cfn, Cfp):
    confusion_matrix = compute_confusion_matrix(predictedLabels, LVAL)
    FN = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    FP = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    DFCDummy = numpy.minimum(p * Cfn, (1 - p) * Cfp)
    DFC = (p * Cfn * FN) + ((1 - p) * Cfp * FP)
    return DFC / DFCDummy

def compute_min_DCF(llr, thresholds, labels, p, Cfn, Cfp): #prendiamo il minor DFC calcolato con ogni threshold, il minor DFC sarebbe il migliore possibile
    DCFarr = []
    thresholds = sorted(thresholds, reverse=True)
    for threshold in thresholds:
        pred = numpy.where(llr > threshold, 1, 0)
        DCFarr.append(compute_binary_normalized_DCF(pred, labels, p, Cfn, Cfp))
    return min(DCFarr)

# Optimize the logistic regression loss
def trainLogReg(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf

# Optimize the weighted logistic regression loss
def trainLogRegPriorWeighted(DTR, LTR, l, pT):
    ZTR = LTR * 2.0 - 1.0  # We do it outside the objective function, since we only need to do it once

    wTar = pT / (ZTR > 0).sum()  # Compute the weights for the two classes
    wNon = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        # Calcolo della loss function con stabilizzazione numerica
        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTar  # Applica i pesi alla loss function
        loss[ZTR < 0] *= wNon

        # Calcolo del gradiente con stabilizzazione numerica
        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G = G * (1.0 - 1.0 / (1.0 + numpy.exp(-numpy.abs(ZTR * s))))  # Stabilizzazione numerica per il gradiente
        G[ZTR > 0] *= wTar  # Applica i pesi al gradiente
        G[ZTR < 0] *= wNon

        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w) ** 2, numpy.hstack([GW, numpy.array(Gb)])

    x,f,_= scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0=numpy.zeros(DTR.shape[0] + 1))
    return x,f

def optimal_bayes_decision(prior, Cfn, Cfp, llrs):
    t = - numpy.log((prior * Cfn)/((1-prior) * Cfp))
    
    P = numpy.full(llrs.shape, 0)
    P[llrs > t] = 1
    P[llrs <= t] = 0

    return P


if __name__ == '__main__':
    
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    lambdas = numpy.logspace(-4, 2, 13)
    th = cost_predictor(0.1, 1, 1)
    thresholds=numpy.linspace(-5,5,100)

    
    actual_dcf_scores = []
    minimum_dcf_scores = []

    pi = 0.1
    
    
    for lamb in lambdas:
        v_opt = trainLogReg(DTR, LTR, lamb)
        w_opt, b_opt = v_opt[:-1], v_opt[-1]

        # Calcola le predizioni sul validation set
        sVal = numpy.dot(w_opt.T, DVAL) + b_opt
        pEmp = (LTR == 1).sum() / LTR.size
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        preds=numpy.where(sValLLR > th, 1, 0)

        
        # Calcola actual DCF e minimum DCF
        act_dcf = compute_binary_normalized_DCF(preds, LVAL, pi, 1, 1)
        min_dcf = compute_min_DCF(sValLLR, thresholds, LVAL, pi, 1, 1)
        
        actual_dcf_scores.append(act_dcf)
        minimum_dcf_scores.append(min_dcf)
        
        print(f"\nLambda: {lamb}")
        print(f"Actual DCF: {act_dcf}")
        print(f"Minimum DCF: {min_dcf}")

    # Tracciare le metriche in funzione di λ
    plt.figure()
    plt.plot(lambdas, actual_dcf_scores, label='Actual DCF', marker='x', color="red")
    plt.plot(lambdas, minimum_dcf_scores, label='Minimum DCF', marker='o', color="blue")
    plt.xscale('log', base=10)
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.legend()
    plt.show()
    


           
    actual_dcf_scores = []
    minimum_dcf_scores = [] 
    
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]


    actual_dcf_scores = []
    minimum_dcf_scores = []

    pi = 0.1
    
    for lamb in lambdas:
        v_opt = trainLogReg(DTR_reduced, LTR_reduced, lamb)
        w_opt, b_opt = v_opt[:-1], v_opt[-1]

        # Calcola le predizioni sul validation set
        sVal = numpy.dot(w_opt.T, DVAL) + b_opt
        pEmp = (LTR_reduced == 1).sum() / LTR_reduced.size
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        preds=numpy.where(sValLLR > th, 1, 0)

        
        # Calcola actual DCF e minimum DCF
        act_dcf = compute_binary_normalized_DCF(preds, LVAL, pi, 1, 1)
        min_dcf = compute_min_DCF(sValLLR, thresholds, LVAL, pi, 1, 1)
        
        actual_dcf_scores.append(act_dcf)
        minimum_dcf_scores.append(min_dcf)
        
        print(f"\nLambda: {lamb}")
        print(f"Actual DCF: {act_dcf}")
        print(f"Minimum DCF: {min_dcf}")
    
    plt.figure()
    plt.plot(lambdas, actual_dcf_scores, label='Actual DCF', marker='x', color="red")
    plt.plot(lambdas, minimum_dcf_scores, label='Minimum DCF', marker='o', color="blue")
    plt.xscale('log', base=10)
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.legend()
    plt.show()
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    actual_dcf_scores_weighted = []
    minimum_dcf_scores_weighted = []
    
    for lamb in lambdas:
        # Train the prior-weighted logistic regression model
        x, f = trainLogRegPriorWeighted(DTR, LTR, lamb, pi)
        w, b = x[:-1], x[-1]
    
        # Calculate predictions on the validation set
        sVal = numpy.dot(w.T, DVAL) + b - numpy.log(pi / (1 - pi))
        pEmp = numpy.mean(LTR == 1)
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        preds = numpy.where(sValLLR > th, 1, 0)
    
        # Calculate actual DCF and minimum DCF
        act_dcf = compute_binary_normalized_DCF(preds, LVAL, pi, 1, 1)
        min_dcf = compute_min_DCF(sValLLR, thresholds, LVAL, pi, 1, 1)
    
        actual_dcf_scores_weighted.append(act_dcf)
        minimum_dcf_scores_weighted.append(min_dcf)
    
        # Print results for the current lambda
        print(f"\nPrior-Weighted Logistic Regression (pT {pi}) - Lambda: {lamb}")
        print(f"Actual DCF: {act_dcf}")
        print(f"Minimum DCF: {min_dcf}")
    
    # Plotting the results
    plt.figure()
    plt.plot(lambdas, actual_dcf_scores_weighted, label='Actual DCF', marker='x', color="red")
    plt.plot(lambdas, minimum_dcf_scores_weighted, label='Minimum DCF', marker='o', color="blue")
    plt.xscale('log', base=10)
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title("Prior-Weighted Logistic Regression")
    plt.legend()
    plt.show()

    
    

    actual_dcf_scores_quadratic = []
    minimum_dcf_scores_quadratic = []
    DTR_expanded = expand_quadratic_features(DTR)
    DVAL_expanded = expand_quadratic_features(DVAL)
    for lamb in lambdas:
       v_opt = trainLogReg(DTR_expanded, LTR, lamb)
       w_opt, b_opt = v_opt[:-1], v_opt[-1]
       sVal = numpy.dot(w_opt.T, DVAL_expanded) + b_opt
       pEmp = (LTR == 1).sum() / LTR.size
       sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
       preds=numpy.where(sValLLR > th, 1, 0)


       # Calculate actual DCF and minimum DCF
       act_dcf = compute_binary_normalized_DCF(preds, LVAL, pi, 1, 1)
       min_dcf = compute_min_DCF(sValLLR, thresholds, LVAL, pi, 1, 1)

       actual_dcf_scores_quadratic.append(act_dcf)
       minimum_dcf_scores_quadratic.append(min_dcf)

       print(f"\nPrior-Weighted Logistic Regression (pT {pi}) - Lambda: {lamb}")
       print(f"Actual DCF: {act_dcf}")
       print(f"Minimum DCF: {min_dcf}")

    plt.figure()
    plt.plot(lambdas, actual_dcf_scores_quadratic, label='Actual DCF', marker='x', color="red")
    plt.plot(lambdas, minimum_dcf_scores_quadratic, label='Minimum DCF', marker='o', color="blue")
    plt.xscale('log', base=10)
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title("Quadratic Logistic Regression")
    plt.legend()
    plt.show()
           
    

    
    
    preprocessing_methods = [
        ("Centering", center_data),
        ("Z-normalization", z_normalize_data),
        ("Whitening", whiten_data),
        ("PCA (m=1)", lambda x, y: apply_PCA(x, y, 1)),
        ("PCA (m=2)", lambda x, y: apply_PCA(x, y, 2)),
        ("PCA (m=3)", lambda x, y: apply_PCA(x, y, 3)),
        ("PCA (m=4)", lambda x, y: apply_PCA(x, y, 4)),
        ("PCA (m=5)", lambda x, y: apply_PCA(x, y, 5)),
    ]

    for method_name, preprocess_func in preprocessing_methods:
        print(f"\nAnalyzing {method_name}")
        DTR_p, DVAL_p = preprocess_func(DTR, DVAL)
        
        min_dcf_array = []
        act_dcf_array = []
        
        for lamb in lambdas:
            v_opt = trainLogReg(DTR_p, LTR, lamb)
            w, b = v_opt[:-1], v_opt[-1]  # Extract weights and bias correctly
            sVal = numpy.dot(w.T, DVAL_p) + b
            pEmp = (LTR == 1).sum() / LTR.size
            sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
            preds=numpy.where(sValLLR > th, 1, 0)

            PVAL = optimal_bayes_decision(pi, 1, 1, sValLLR)
            act_dcf = compute_binary_normalized_DCF(preds, LVAL, pi, 1, 1)
            min_dcf = compute_min_DCF(sValLLR, thresholds, LVAL, pi, 1, 1)
            print(f'λ = {lamb:.4e}, minDCF = {min_dcf:.4f}, actDCF = {act_dcf:.4f}')
            min_dcf_array.append(min_dcf)
            act_dcf_array.append(act_dcf)
        
        plt.figure()
        plt.xscale('log', base=10)
        plt.plot(lambdas, act_dcf_array, label='Actual DCF', marker='x', color="red")
        plt.plot(lambdas, min_dcf_array, label='Minimum DCF', marker='o', color="blue")
        plt.xlabel("λ value")
        plt.ylabel("DCF value")
        plt.title(f"DCF vs λ [{method_name}]")
        plt.legend()
        plt.show()
        
    logreg_mindcf = []
    weighted_logreg_mindcf = []
    quadratic_logreg_mindcf = []
    
    DTR_expanded = expand_quadratic_features(DTR)
    DVAL_expanded = expand_quadratic_features(DVAL)
    
    for lamb in lambdas:
        
        # Logistic Regression
        v_opt = trainLogReg(DTR, LTR, lamb)
        w_opt, b_opt = v_opt[:-1], v_opt[-1]
        sVal = numpy.dot(w_opt.T, DVAL) + b_opt
        pEmp = numpy.mean(LTR == 1)
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        logreg_mindcf.append(compute_min_DCF(sValLLR, thresholds, LVAL, pi, 1, 1))
        
        # Prior Weighted Logistic Regression
        x, f = trainLogRegPriorWeighted(DTR, LTR, lamb, pi)
        w, b = x[:-1], x[-1]
        sVal = numpy.dot(w.T, DVAL) + b - numpy.log(pi / (1 - pi))
        weighted_sllr = sVal - numpy.log(pEmp / (1 - pEmp))
        weighted_logreg_mindcf.append(compute_min_DCF(weighted_sllr, thresholds, LVAL, pi, 1, 1))
        
        # Quadratic Logistic Regression
        v_opt = trainLogReg(DTR_expanded, LTR, lamb)
        w_opt, b_opt = v_opt[:-1], v_opt[-1]
        sVal = numpy.dot(w_opt.T, DVAL_expanded) + b_opt
        quadratic_sllr = sVal - numpy.log(pEmp / (1 - pEmp))
        quadratic_logreg_mindcf.append(compute_min_DCF(quadratic_sllr, thresholds, LVAL, pi, 1, 1))
    
    print("min DCF Logistic Regression %f" % min(logreg_mindcf))
    print("min DCF Prior Weighted Logistic Regression %f" % min(weighted_logreg_mindcf))
    print("min DCF Quadratic Logistic Regression %f" % min(quadratic_logreg_mindcf))
    
    
    

