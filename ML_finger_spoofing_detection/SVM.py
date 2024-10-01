#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:55:26 2024

@author: LucaBergamini
"""

import numpy
import scipy
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

def func_with_grad(x):
    y, z = x
    f_val = (y + 3)**2 + numpy.sin(y) + (z + 1)**2
    grad = numpy.array([2*(y + 3) + numpy.cos(y), 2*(z + 1)])
    return f_val, grad

def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    ZTR = LTR * 2.0 - 1.0
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    def compute_fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(compute_fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    def compute_primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    compute_primalLoss, dualLoss = compute_primalLoss(w_hat), -compute_fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, compute_primalLoss, dualLoss, compute_primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):
    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def compute_fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(compute_fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -compute_fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def compute_fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return compute_fScore # we directly return the function to score a matrix of test samples

def optimal_bayes_decision(prior, Cfn, Cfp, llrs):
    t = - numpy.log((prior * Cfn)/((1-prior) * Cfp))
    
    P = numpy.full(llrs.shape, 0)
    P[llrs > t] = 1
    P[llrs <= t] = 0

    return P

def center_data(DTR, DTE):
    mu = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mu
    DTE_centered = DTE - mu
    return DTR_centered, DTE_centered

if __name__ == "__main__":
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    C_values = numpy.logspace(-5, 0, 11)
    pi = 0.1

    # Linear SVM training (già nel tuo codice)
    K_value = 1

    min_dcf_array = []
    act_dcf_array = []
    for C in C_values:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K_value)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        
        min_dcf = compute_min_DCF(SVAL, SVAL, LVAL, pi, 1, 1)
        act_dcf = compute_empirical_Bayes_risk_optimal(SVAL, LVAL, pi, 1, 1)
        min_dcf_array.append(min_dcf)
        act_dcf_array.append(act_dcf)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % act_dcf)
    

    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(C_values, act_dcf_array, label='Actual DCF', marker='x', color="red")
    plt.plot(C_values, min_dcf_array, label='Minimum DCF', marker='o', color="blue")    
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title('Performance of Linear SVM with different C values')
    plt.legend()
    plt.show()   
        
        
    DTR_centered, DVAL_centered = center_data(DTR, DVAL)

    min_dcf_array = []
    act_dcf_array = []
    
    
    
    C_values = numpy.logspace(-5, 0, 11)
    for C in C_values:
        w, b = train_dual_SVM_linear(DTR_centered, LTR, C, K_value)
        SVAL = (vrow(w) @ DVAL_centered + b).ravel()
        min_dcf = compute_min_DCF(SVAL, SVAL, LVAL, pi, 1, 1)
        act_dcf = compute_empirical_Bayes_risk_optimal(SVAL, LVAL, pi, 1, 1)
        min_dcf_array.append(min_dcf)
        act_dcf_array.append(act_dcf)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % act_dcf)
    

    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(C_values, act_dcf_array, label='Actual DCF', marker='x', color="red")
    plt.plot(C_values, min_dcf_array, label='Minimum DCF', marker='o', color="blue")   
    plt.xlabel("C value")
    plt.ylabel("DCF value")
    plt.title("DCF vs C [Linear SVM, centered data]")
    plt.legend()
    plt.show()  
    
    
    
    d = 2 #polynomial degree
    c = 1 #polynomial kernel function hyper-parameter 
    eps = 0 #0 value becaus c = 1 already account for having K = 1 as additional component to each training sample
    min_dcf_array = []
    act_dcf_array = []
    for C in C_values:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, polyKernel(d, c), eps) #returns the fScore function built
        SVAL = fScore(DVAL)
        min_dcf = compute_min_DCF(SVAL, SVAL, LVAL, pi, 1, 1)
        act_dcf = compute_empirical_Bayes_risk_optimal(SVAL, LVAL, pi, 1, 1)
        min_dcf_array.append(min_dcf)
        act_dcf_array.append(act_dcf)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % act_dcf)
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(C_values, act_dcf_array, label='Actual DCF', marker='x', color="red")
    plt.plot(C_values, min_dcf_array, label='Minimum DCF', marker='o', color="blue")
    plt.xlabel("C value")
    plt.ylabel("DCF value")
    plt.title("DCF vs C [Non-linear SVM, polynomial Kernel]")
    plt.legend()
    plt.show() 
    
    

    gamma_values = [(numpy.exp(-4), "b", r"$\gamma=e^{-4}$") , (numpy.exp(-3), "r", r"$\gamma=e^{-3}$"), (numpy.exp(-2), "m", r"$\gamma=e^{-2}$"), (numpy.exp(-1), "y", r"$\gamma=e^{-1}$")]
    C_values = numpy.logspace(-3, 2, 11) 
    eps = 1 # RBF kernel does not account for the bias term
    for gamma in gamma_values:
        print("Gamma = ", gamma)
        min_dcf_array = []
        act_dcf_array = []
        for C in C_values:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(gamma[0]), eps) #returns the fScore function built
            SVAL = fScore(DVAL)
            min_dcf = compute_min_DCF(SVAL, SVAL, LVAL, pi, 1, 1)
            act_dcf = compute_empirical_Bayes_risk_optimal(SVAL, LVAL, pi, 1, 1)
            min_dcf_array.append(min_dcf)
            act_dcf_array.append(act_dcf)
            print ('minDCF - pT = 0.1: %.4f' % min_dcf)
            print ('actDCF - pT = 0.1: %.4f' % act_dcf)
        plt.xscale('log', base=10)
        plt.plot(C_values, min_dcf_array, label=r"min DCF "+gamma[2], color=gamma[1], marker='o')
        plt.plot(C_values, act_dcf_array, label=r"actual DCF "+gamma[2], color=gamma[1], marker="x", linestyle='--')    
    plt.title('DCF vs C [Non-linear SVM, RBF Kernel]')
    plt.grid(True)
    plt.legend()
    plt.xlabel("C value")
    plt.ylabel("DCF value")
    plt.show()


    #Non-linear SVM, polynomial kernel, d = 4

    C_values = numpy.logspace(-5, 0, 11)
    d = 3 #polynomial degree
    c = 1 #polynomial kernel function hyper-parameter 
    eps = 0 #0 value becaus c = 1 already account for having K = 1 as additional component to each training sample
    min_dcf_array = []
    act_dcf_array = []
    for C in C_values:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, polyKernel(d, c), eps) #returns the fScore function built
        SVAL = fScore(DVAL)
        min_dcf = compute_min_DCF(SVAL, SVAL, LVAL, pi, 1, 1)
        act_dcf = compute_empirical_Bayes_risk_optimal(SVAL, LVAL, pi, 1, 1)
        min_dcf_array.append(min_dcf)
        act_dcf_array.append(act_dcf)
        print ('minDCF - pT = 0.1: %.4f' % min_dcf)
        print ('actDCF - pT = 0.1: %.4f' % act_dcf)
    
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(C_values, act_dcf_array, label='Actual DCF', marker='x', color="red")
    plt.plot(C_values, min_dcf_array, label='Minimum DCF', marker='o', color="blue")
    plt.xlabel("C value")
    plt.ylabel("DCF value")
    plt.title("DCF vs C [Non-linear SVM, polynomial Kernel]")
    plt.legend()
    plt.show()
