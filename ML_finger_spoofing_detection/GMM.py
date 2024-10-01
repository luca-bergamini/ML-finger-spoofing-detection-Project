#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 02:01:56 2024

@author: LucaBergamini
"""

import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

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

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

def compute_DCF_bayes_cost_norm(predictedLabels, LVAL, p, Cfn, Cfp):
    confusion_matrix= compute_confusion_matrix(predictedLabels, LVAL)
    FN = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    FP = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    DFCDummy = numpy.minimum(p * Cfn, (1 - p) * Cfp)
    DFC = (p * Cfn * FN) + ((1 - p) * Cfp * FP)
    return DFC / DFCDummy

def compute_empirical_Bayes_risk_optimal(llr, classLabels, prior, Cfn, Cfp):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_DCF_bayes_cost_norm(predictedLabels, classLabels, prior, Cfn, Cfp)

def logpdf_GAU_ND (X , mu , C):
    Y = []
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu=vcol(mu)
    t1 = X.shape[0]* numpy.log(2 * numpy.pi)
    t2 = numpy.linalg.slogdet(C)[1]
    for x in X.T:
        x = vcol(x)
        t3 = numpy.dot(numpy.dot((x-mu).T,numpy.linalg.inv(C)),(x-mu))[0,0]
        Y.append(-0.5*(t1+t2+t3))
    return numpy.array(Y)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GMM(X, gmm):
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)

    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):
    U, s, Vh = numpy.linalg.svd(C)
    s[s < psi] = psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

def train_GMM_EM_Iteration(X, gmm, covType='Full', psiEig=None):
    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    # E-step
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)

    S = numpy.vstack(S)  # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0)  # Compute marginal for samples f(x_i)

    gammaAllComponents = numpy.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)):
        # Compute statistics:
        gamma = gammaAllComponents[gIdx]  # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))  # Exploit broadcasting to compute the sum
        S = (vrow(gamma) * X) @ X.T
        muUpd = F / Z
        CUpd = S / Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd = CUpd * numpy.eye(X.shape[0])  # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]

    return gmmUpd

# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
def train_GMM_EM(X, gmm, covType='Full', psiEig=None, epsLLAverage=1e-6):
    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType=covType, psiEig=psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))
    return gmm

def split_GMM_LBG(gmm, alpha=0.1):
    gmmOut = []
    print('LBG - going from %d to %d components' % (len(gmm), len(gmm) * 2))
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

# Train a full model using LBG + EM, starting from a single Gaussian model, until we have numComponents components. lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType='Full', psiEig=None, epsLLAverage=1e-6, lbgAlpha=0.1):
    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0])  # We need an initial diagonal GMM to train a diagonal GMM

    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C,psiEig))]  # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)]  # 1-component model

    while len(gmm) < numComponents:
        # Split the components
        print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha)
        print('Average ll after LBG: %.8e' % logpdf_GMM(X,gmm).mean())  # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, epsLLAverage=epsLLAverage)
    return gmm
    
def  logReg(DTR,LTR,l):
    def logreg_obj(v):
        w,b = v[0:-1], v[-1]
        z = 2*LTR-1
        reg_term = l/2*numpy.linalg.norm(w)**2
        exponent = -z*(numpy.dot(w.T,DTR)+b)
        sum = numpy.logaddexp(0,exponent).sum()
        return reg_term + sum/DTR.shape[1]
    x,f,_ = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    return x,f

def quadratic_features(X):
    n_samples = X.shape[1]
    quadratic_features = []

    for i in range(n_samples):
        sample = X[:, i]
        quadratic_sample = []
        for j in range(len(sample)):
            for k in range(j, len(sample)):
                quadratic_sample.append(sample[j] * sample[k])
        quadratic_features.append(quadratic_sample)

    quadratic_features = numpy.array(quadratic_features).T
    return numpy.vstack((X, quadratic_features))

def compute_H_rbf(DTR, LTR, K, gamma):
    # Compute the label vector
    Z = numpy.where(LTR == 1, 1, -1)

    # Compute the H matrix using RBF kernel function
    H = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            H[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = numpy.outer(Z, Z) * H

    return H, Z

def SVM_RBF(DTR, LTR, DTE, C, K, gamma):
    H, Z = compute_H_rbf(DTR, LTR, K, gamma)

    # Define the dual objective function
    def dual_func(alpha_values):
        Ha_values = numpy.dot(H, alpha_values.reshape(-1, 1))
        aHa_value = numpy.dot(alpha_values, Ha_values)
        a1_value = alpha_values.sum()
        result = -0.5 * aHa_value + a1_value, (-Ha_values + 1).flatten()
        l = - result[0]
        g = - result[1]
        return l, g

    # Optimize the dual objective function
    optimal_alpha, _, _ = fmin_l_bfgs_b(dual_func,numpy.zeros(DTR.shape[1]),bounds=([(0, C)] * DTR.shape[1]),factr=1.0,maxiter=100000,maxfun=100000,)

    # Compute the kernel matrix outside the function
    kernel_matrix = compute_kernel(DTR, DTE, gamma) + K * K

    Z_row = Z.reshape((1, Z.size))
    # Compute the scores
    computed_scores = numpy.sum(numpy.dot(optimal_alpha * Z_row, kernel_matrix), axis=0)
    return computed_scores.ravel()

def compute_kernel(train_data, test_data, gamma):

    kernel_matrix = numpy.zeros((train_data.shape[1], test_data.shape[1]))
    for i in range(train_data.shape[1]):
        for j in range(test_data.shape[1]):
            kernel_matrix[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(train_data[:, i] - test_data[:, j]) ** 2))
    return kernel_matrix

def GMM_cls_iteration(DTR,LTR,DVAL,numC,covType):
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType,  psiEig = 0.01)

    sllr = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    return sllr

if __name__ == "__main__":
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    p, Cfn, Cfp= 0.1, 1, 1
    
    values = numpy.array([2,4,8,16,32])
    th = cost_predictor(p, Cfn, Cfp)
    act_array=[]
    min_array=[]
    
    for numC in values:
        sllr= GMM_cls_iteration(DTR,LTR,DVAL,numC,"full")
        predictions=numpy.where(sllr > th, 1, 0)
        
        min_dcf = compute_min_DCF(sllr, sllr, LVAL, p, Cfn, Cfp)
        act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, 1, 1)
        act_array.append(act_dcf)
        min_array.append(min_dcf)
        
        
    plt.plot(values, act_array, label='DCF', color='r')
    plt.plot(values, min_array, label='minDCF', color='b')
    plt.scatter(values, act_array, color='r')
    plt.scatter(values, min_array, color='b')
    plt.legend()
    plt.title("GMM")
    plt.show()


    act_array=[]
    min_array=[]
    
    for numC in values:
        sllr= GMM_cls_iteration(DTR,LTR,DVAL,numC,"diagonal")
        predictions=numpy.where(sllr > th, 1, 0)
        
        min_dcf = compute_min_DCF(sllr, sllr, LVAL, p, Cfn, Cfp)
        act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, 1, 1)
        act_array.append(act_dcf)
        min_array.append(min_dcf)
        
        
    plt.plot(values, act_array, label='DCF', color='r')
    plt.plot(values, min_array, label='minDCF', color='b')
    plt.scatter(values, act_array, color='r')
    plt.scatter(values, min_array, color='b')
    plt.legend()
    plt.title("GMM diagonal")
    plt.show()

    quadratic_DTR=quadratic_features(DTR)
    quadratic_DVAL=quadratic_features(DVAL)
    x,f=logReg(quadratic_DTR,LTR,0.01)
    w, b = x[0:-1], x[-1]
    piemp = numpy.mean(LTR == 1)
    sllr = numpy.dot(w.T,quadratic_DVAL) + b - numpy.log(piemp / (1 - piemp))
    
    x = -4
    y = 4
    n = 31
    
    effPriorLogOdds = numpy.linspace(x, y, n)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    act_array=[]
    min_array=[]
    

    for p in pi:
        th = cost_predictor(p,Cfn,Cfp)
        predictions = numpy.where(sllr > th, 1, 0)
        min_dcf = compute_min_DCF(sllr, sllr, LVAL, p, Cfn, Cfp)
        act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, 1, 1)
        act_array.append(act_dcf)
        min_array.append(min_dcf)

    plt.plot(effPriorLogOdds, act_array, label='DCF', color ='r')
    plt.plot(effPriorLogOdds, min_array, label='minDCF', color ='b')
    plt.ylim([0, 1.1])
    plt.xlim([x,y])
    plt.legend(loc='lower left')
    plt.title("Quadratic Logistic Regression Bayes Error Plot")
    plt.show()
    
    
    scores= SVM_RBF(DTR,LTR,DVAL,100,1.0,0.1)
    
    
    x = -4
    y = 4
    n = 31
    
    effPriorLogOdds = numpy.linspace(x, y, n)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    act_array=[]
    min_array=[]
    

    for p in pi:
        th = cost_predictor(p,Cfn,Cfp)
        predictions = numpy.where(scores > th, 1, 0)
        min_dcf = compute_min_DCF(scores, scores, LVAL, p, Cfn, Cfp)
        act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, 1, 1)
        act_array.append(act_dcf)
        min_array.append(min_dcf)

    plt.plot(effPriorLogOdds, act_array, label='DCF', color ='r')
    plt.plot(effPriorLogOdds, min_array, label='minDCF', color ='b')
    plt.ylim([0, 1.1])
    plt.xlim([x,y])
    plt.legend(loc='lower left')
    plt.title("SVM RBF Bayes Error Plot")
    plt.show()
    
    
    
    sllr = GMM_cls_iteration(DTR,LTR,DVAL,8,"diagonal")
    x = -4
    y = 4
    n = 31
    
    effPriorLogOdds = numpy.linspace(x, y, n)
    pi=1/(1 + numpy.exp(-effPriorLogOdds))
    act_array=[]
    min_array=[]
    

    for p in pi:
        th = cost_predictor(p,Cfn,Cfp)
        predictions = numpy.where(sllr > th, 1, 0)
        min_dcf = compute_min_DCF(sllr, sllr, LVAL, p, Cfn, Cfp)
        act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, 1, 1)
        act_array.append(act_dcf)
        min_array.append(min_dcf)

    plt.plot(effPriorLogOdds, act_array, label='DCF', color ='r')
    plt.plot(effPriorLogOdds, min_array, label='minDCF', color ='b')
    plt.ylim([0, 1.1])
    plt.xlim([x,y])
    plt.legend(loc='lower left')
    plt.title("GMM diagonal Bayes Error Plot")
    plt.show()


