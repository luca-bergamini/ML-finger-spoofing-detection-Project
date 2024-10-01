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

def weightedLogReg(DTR, LTR, l, pT):
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

def weightedLogReg_sllr(DTR, LTR, l, DVAL, p):
    x,f = weightedLogReg(DTR,LTR,l,p)
    w, b = x[0:-1], x[-1]
    sllr = numpy.dot(w.T, DVAL) + b - numpy.log(p/(1 - p))
    return sllr

def logpdf_GAU_ND (X, mu, C):
    Y = []
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = vcol(mu)
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

def extract_train_val_folds_from_ary(X, idx, KFOLD):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def Kfold_calibration(scores, labels, p, Cfn, Cfp):
    KFOLD = 5
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    thresh = cost_predictor(p, Cfn, Cfp)
    predictions = numpy.where(scores > thresh,1,0)
    dcf = compute_DCF_bayes_cost_norm(predictions, labels, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(scores, scores, labels, p, Cfn, Cfp)
    print(dcf, min_dcf)
    print()

    for prior in priors :
        calibrated_scores = []
        labels_calibration = []
        for foldIdx in range(KFOLD):
            SCAL, SVAL = extract_train_val_folds_from_ary(scores, foldIdx ,KFOLD)
            LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)

            calibrated_SVAL = weightedLogReg_sllr(vrow(SCAL), LCAL, 0, vrow(SVAL), prior)

            calibrated_scores.append(calibrated_SVAL)
            labels_calibration.append(LVAL)

        calibrated_scores = numpy.hstack(calibrated_scores)
        labels_calibrated = numpy.hstack(labels_calibration)

        calibrated_predictions = numpy.where(calibrated_scores > thresh, 1, 0)
        
        dcf = compute_DCF_bayes_cost_norm(calibrated_predictions, labels_calibrated, p, Cfn, Cfp)
        min_dcf = compute_min_DCF(calibrated_scores, calibrated_scores, labels_calibrated, p, Cfn, Cfp)
        print(prior, f"{dcf:.4f}")

def  logReg(DTR, LTR, l):
    def logreg_obj(v):
        w,b = v[0:-1], v[-1]
        z = 2*LTR-1
        reg_term = l/2*numpy.linalg.norm(w)**2
        exponent = -z*(numpy.dot(w.T,DTR)+b)
        sum = numpy.logaddexp(0,exponent).sum()
        return reg_term + sum/DTR.shape[1]
    x,f,_ = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    return x,f


def qaudraticlogreg_sllr(DTR, LTR, l, DVAL):
    DTR_expanded = expand_quadratic_features(DTR)
    DVAL_expanded = expand_quadratic_features(DVAL)
    x,f = logReg(DTR_expanded, LTR, l)
    w, b = x[0:-1], x[-1]
    piemp = numpy.mean(LTR == 1)
    sllr = numpy.dot(w.T, DVAL_expanded) + b - numpy.log(piemp / (1 - piemp))
    return sllr


def fusion(scores1, scores2, scores3, labels, p, Cfn, Cfp):
    KFOLD = 5
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    thresh = cost_predictor(p, Cfn, Cfp)

    for prior in priors :
        calibrated_scores = []
        labels_calibration = []
        for foldIdx in range(KFOLD):
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores1, foldIdx ,KFOLD)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores2, foldIdx, KFOLD)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores3, foldIdx, KFOLD)
            LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)

            SCAL = numpy.vstack([SCAL1, SCAL2, SCAL3])
            SVAL = numpy.vstack([SVAL1, SVAL2, SVAL3])
            calibrated_SVAL = weightedLogReg_sllr(SCAL, LCAL, 0, SVAL, prior)

            calibrated_scores.append(calibrated_SVAL)
            labels_calibration.append(LVAL)

        calibrated_scores = numpy.hstack(calibrated_scores)
        labels_calibrated = numpy.hstack(labels_calibration)

        calibrated_predictions = numpy.where(calibrated_scores > thresh,1,0)

        dcf = compute_DCF_bayes_cost_norm(calibrated_predictions, labels_calibrated, p, Cfn,Cfp)
        print(prior, f"{dcf:.4f}")
        
def logreg_sllr(DTR, LTR, l, DVAL):
    x,f = logReg(DTR, LTR, l)
    w, b = x[0:-1], x[-1]
    piemp = numpy.mean(LTR == 1)
    sllr = numpy.dot(w.T, DVAL) + b - numpy.log(piemp / (1 - piemp))
    return sllr
        
def get_fusion_scores(scores, prior, labels):
    calibrated_scores = weightedLogReg_sllr(scores, labels, 0, scores, prior)
    return calibrated_scores

def get_calibrated_scores(scores, prior, labels):
    calibrated_scores = weightedLogReg_sllr(vrow(scores), labels, 0, vrow(scores), prior)
    return calibrated_scores

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

def smooth_covariance_matrix(C, psi):
    U, s, Vh = numpy.linalg.svd(C)
    s[s < psi] = psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

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

def logpdf_GMM(X, gmm):
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)

    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def GMM_cls_iteration(DTR,LTR,DVAL,numC,covType):
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType,  psiEig = 0.01)

    sllr = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    return sllr

if __name__ == "__main__":
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    logreg = numpy.load("quadratic_logreg_sllr.npy")
    SVM = numpy.load("SVM_RBF_scores.npy")
    GMM = numpy.load("GMM_diagoanl_sllr.npy")
    
    Cfn = 1
    Cfp = 1
    
    p = 0.1
    
    Kfold_calibration(logreg,LVAL,p,Cfn,Cfp)
    Kfold_calibration(SVM,LVAL,p,Cfn,Cfp)
    Kfold_calibration(GMM,LVAL,p,Cfn,Cfp)
    
    prior_cal_logreg =  0.4
    prior_cal_SVM_RBF = 0.9
    prior_cal_diagGMM = 0.8
    
    
    quad_logreg_cal_sllr = get_calibrated_scores(logreg, prior_cal_logreg, LVAL)
    SVM_RBF_cal_scores = get_calibrated_scores(SVM, prior_cal_SVM_RBF, LVAL)
    GMM_diagonal_cal_sllr = get_calibrated_scores(GMM, prior_cal_diagGMM, LVAL)
    
    th = cost_predictor(p, Cfn, Cfp)
    predictions = numpy.where(quad_logreg_cal_sllr > th, 1, 0)
    act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(quad_logreg_cal_sllr, quad_logreg_cal_sllr, LVAL, p, Cfn, Cfp)
    print(act_dcf)
    print(min_dcf)
    
    predictions = numpy.where(SVM_RBF_cal_scores > th, 1, 0)
    act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(SVM_RBF_cal_scores, SVM_RBF_cal_scores, LVAL, p, Cfn, Cfp)
    print(act_dcf)
    print(min_dcf)
    
    
    predictions = numpy.where(GMM_diagonal_cal_sllr > th, 1, 0)
    act_dcf = compute_DCF_bayes_cost_norm(predictions, LVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(GMM_diagonal_cal_sllr, GMM_diagonal_cal_sllr, LVAL, p, Cfn, Cfp)
    print(act_dcf)
    print(min_dcf)

    x,y,n = -4, 4, 31
    effPriorLogOdds = numpy.linspace(x, y, n)
    pi = 1/(1 + numpy.exp(-effPriorLogOdds))
    logreg_dcf = []
    logreg_mindcf = []
    SVM_dcf = []
    SVM_mindcf = []
    GMM_dcf = []
    GMM_mindcf = []
    
    for i in pi:
        thresh = cost_predictor(i, Cfn, Cfp)
        logreg_p = numpy.where(quad_logreg_cal_sllr > thresh, 1, 0)
        
        min_logreg_dcf = compute_min_DCF(quad_logreg_cal_sllr, quad_logreg_cal_sllr, LVAL, i, Cfn, Cfp)
        act_logreg_dcf = compute_DCF_bayes_cost_norm(logreg_p, LVAL, i, Cfn, Cfp)

        logreg_dcf.append(act_logreg_dcf)
        logreg_mindcf.append(min_logreg_dcf)

        SVM_p = numpy.where(SVM_RBF_cal_scores > thresh, 1, 0)
        min_svm_dcf = compute_min_DCF(SVM_RBF_cal_scores, SVM_RBF_cal_scores, LVAL, i, Cfn, Cfp)
        act_svm_dcf = compute_DCF_bayes_cost_norm(SVM_p, LVAL, i, Cfn, Cfp)

        SVM_dcf.append(act_svm_dcf)
        SVM_mindcf.append(min_svm_dcf)

        GMM_p = numpy.where(GMM_diagonal_cal_sllr > thresh, 1, 0)
        min_GMM_dcf = compute_min_DCF(GMM_diagonal_cal_sllr, GMM_diagonal_cal_sllr, LVAL, i, Cfn, Cfp)
        act_GMM_dcf = compute_DCF_bayes_cost_norm(GMM_p, LVAL, i, Cfn, Cfp)

        GMM_dcf.append(act_GMM_dcf)
        GMM_mindcf.append(min_GMM_dcf)

    plt.plot(effPriorLogOdds, logreg_dcf, label='quadratic logreg DCF', color ='r')
    plt.plot(effPriorLogOdds, logreg_mindcf,linestyle='--', label='quadratic logreg minDCF', color ='r')
    plt.plot(effPriorLogOdds, SVM_dcf, label='SVM RBF DCF', color ='b')
    plt.plot(effPriorLogOdds, SVM_mindcf,linestyle='--', label='SVM RBF minDCF', color ='b')
    plt.plot(effPriorLogOdds, GMM_dcf, label='diagonal GMM DCF', color ='g')
    plt.plot(effPriorLogOdds, GMM_mindcf,linestyle='--', label='diagonal GMM minDCF', color ='g')

    plt.ylim([0, 1.1])
    plt.xlim([x, y])
    plt.legend(loc='upper left')
    plt.title("calibrated models Bayes error plot")
    plt.savefig("calibrated models Bayes error plot")
    plt.show()
    
    

    fusion(logreg, SVM, GMM, LVAL, p, Cfn, Cfp)

    p, Cfn, Cfp = 0.1, 1, 1
    prior_fusion = 0.8
    th = cost_predictor(p, Cfn, Cfp)

    fusion_scores = numpy.vstack([logreg,SVM,GMM])
    fusion_cal_sllr = get_fusion_scores(fusion_scores, prior_fusion, LVAL)
    fusion_cal_sllr_predictions = numpy.where(fusion_cal_sllr > th, 1, 0)
    act_dcf = compute_DCF_bayes_cost_norm(fusion_cal_sllr_predictions, LVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(fusion_cal_sllr, fusion_cal_sllr, LVAL, p, Cfn, Cfp)
    
    print(f"act_dcf: {act_dcf:.4f}")
    print(f"min_dcf: {min_dcf:.4f}")
    
    DEVAL, LEVAL= load("evalData.txt")
    sllr_logreg = qaudraticlogreg_sllr(DTR, LTR, 0.01, DEVAL)
    numpy.save("logreg_eval", sllr_logreg)
    sllr_SVM= SVM_RBF(DTR, LTR, DEVAL, 100, 1.0, 0.1)
    numpy.save("SVM_eval", sllr_SVM)
    sllr_GMM = GMM_cls_iteration(DTR, LTR, DEVAL, 8, "diagonal")
    numpy.save("GMM_eval", sllr_GMM)

    sllr_logreg = numpy.load("logreg_eval.npy")
    sllr_SVM = numpy.load("SVM_eval.npy")
    sllr_GMM = numpy.load("GMM_eval.npy")

    x,_= weightedLogReg(fusion_scores, LVAL, 0, 0.8)
    w, b = x[0:-1], x[-1]
    sllr_fusion = numpy.dot(w.T, numpy.vstack([sllr_logreg,sllr_SVM,sllr_GMM])) + b - numpy.log(0.8/(1 - 0.8))
    predictions_fusion = numpy.where(sllr_fusion > th, 1, 0)
    error = numpy.sum(predictions_fusion != LEVAL)
    
    act_dcf = compute_DCF_bayes_cost_norm(predictions_fusion, LEVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(sllr_fusion, sllr_fusion, LEVAL, p, Cfn, Cfp)
    print(error,act_dcf,min_dcf)

    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi= 1/(1 + numpy.exp(-effPriorLogOdds))
    dcf = []
    mindcf = []
    
    for i in pi:
        thresh= cost_predictor(i, Cfn, Cfp)
        predictions = numpy.where(sllr_fusion > thresh, 1, 0)
        
        act_dcf = compute_DCF_bayes_cost_norm(predictions, LEVAL, i, Cfn, Cfp)
        min_dcf = compute_min_DCF(sllr_fusion, sllr_fusion, LEVAL, i, Cfn, Cfp)
        dcf.append(act_dcf)
        mindcf.append(min_dcf)
        
    plt.plot(effPriorLogOdds, dcf, label='DCF', color ='r')
    plt.plot(effPriorLogOdds, mindcf, label='minDCF', color ='b')
    plt.ylim([0, 1.1])
    plt.legend(loc='lower left')
    plt.title("Set fusion Bayes error plot")
    plt.show()

    p, Cfn, Cfp = 0.1, 1, 1

    x,_ = weightedLogReg(vrow(logreg), LVAL, 0, 0.4)
    w, b = x[0:-1], x[-1]
    sllr_logreg = numpy.dot(w.T, vrow(sllr_logreg)) + b - numpy.log(0.4/(1 - 0.4))
    
    act_dcf = compute_empirical_Bayes_risk_optimal(sllr_logreg, LEVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(sllr_logreg, sllr_logreg, LEVAL, p, Cfn, Cfp)
    
    print(f"logreg act_dcf: {act_dcf:.4f}")
    print(f"logreg min_dcf: {min_dcf:.4f}")
    
    x,_ = weightedLogReg(vrow(SVM), LVAL, 0, 0.9)
    w, b = x[0:-1], x[-1]
    sllr_SVM = numpy.dot(w.T, vrow(sllr_SVM)) + b - numpy.log(0.9/(1 - 0.9))
    
    act_dcf = compute_empirical_Bayes_risk_optimal(sllr_SVM, LEVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(sllr_SVM, sllr_SVM, LEVAL, p, Cfn, Cfp)
    
    print(f"SVM act_dcf: {act_dcf:.4f}")
    print(f"SVM min_dcf: {min_dcf:.4f}")
    
    x,_ = weightedLogReg(vrow(GMM), LVAL, 0, 0.8)
    w, b = x[0:-1], x[-1]
    sllr_GMM = numpy.dot(w.T, vrow(sllr_GMM)) + b - numpy.log(0.8/(1 - 0.8))
    
    act_dcf = compute_empirical_Bayes_risk_optimal(sllr_GMM, LEVAL, p, Cfn, Cfp)
    min_dcf = compute_min_DCF(sllr_GMM, sllr_GMM, LEVAL, p, Cfn, Cfp)
    
    print(f"GMM act_dcf: {act_dcf:.4f}")
    print(f"GMM min_dcf: {min_dcf:.4f}")
    
    effPriorLogOdds = numpy.linspace(-4, 4, 31)
    pi = 1/(1 + numpy.exp(-effPriorLogOdds))
    
    logreg_dcf = []
    logreg_mindcf = []
    SVM_dcf = []
    SVM_mindcf = []
    GMM_dcf = []
    GMM_mindcf = []
    fusion_dcf = []
    fusion_mindcf = []
    
    for i in pi:
        thresh = cost_predictor(i, Cfn, Cfp)

        act_dcf_l = compute_empirical_Bayes_risk_optimal(sllr_logreg, LEVAL, i, Cfn, Cfp)
        min_dcf_l = compute_min_DCF(sllr_logreg, sllr_logreg, LEVAL, i, Cfn, Cfp)
    
        logreg_dcf.append(act_dcf_l)
        logreg_mindcf.append(min_dcf_l)

        act_dcf_s = compute_empirical_Bayes_risk_optimal(sllr_SVM, LEVAL, i, Cfn, Cfp)
        min_dcf_s = compute_min_DCF(sllr_SVM, sllr_SVM, LEVAL, i, Cfn, Cfp)

        SVM_dcf.append(act_dcf_s)
        SVM_mindcf.append(min_dcf_s)

        act_dcf_g = compute_empirical_Bayes_risk_optimal(sllr_GMM, LEVAL, i, Cfn, Cfp)
        min_dcf_g = compute_min_DCF(sllr_GMM, sllr_GMM, LEVAL, i, Cfn, Cfp)

        GMM_dcf.append(act_dcf_g)
        GMM_mindcf.append(min_dcf_g)

        act_dcf_f = compute_empirical_Bayes_risk_optimal(sllr_fusion, LEVAL, i, Cfn, Cfp)
        min_dcf_f = compute_min_DCF(sllr_fusion, sllr_fusion, LEVAL, i, Cfn, Cfp)

        fusion_dcf.append(act_dcf_f)
        fusion_mindcf.append(min_dcf_f)
    
    plt.plot(effPriorLogOdds, logreg_dcf, label='quadratic logreg DCF')
    plt.plot(effPriorLogOdds, logreg_mindcf,linestyle='--', label='quadratic logreg minDCF')
    plt.plot(effPriorLogOdds, SVM_dcf, label='SVM RBF DCF')
    plt.plot(effPriorLogOdds, SVM_mindcf,linestyle='--', label='SVM RBF minDCF')
    plt.plot(effPriorLogOdds, GMM_dcf, label='diagonal GMM DCF')
    plt.plot(effPriorLogOdds, GMM_mindcf,linestyle='--', label='diagonal GMM minDCF')
    plt.plot(effPriorLogOdds, fusion_dcf, label='fusion DCF')
    plt.plot(effPriorLogOdds, fusion_mindcf,linestyle='--', label='fusion minDCF')
    
    plt.legend(loc='upper left')
    plt.title("final models Bayes error plot")
    plt.savefig("final Bayes error plot")
    plt.show()
    
    D, L = load("trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    x = numpy.logspace(-4, 2, 13)
    mindcf_logreg = []
    mindcf_weightedlogreg = []
    mindcf_quadraticlogreg = []
    
    for l in x:
        sllr1 = logreg_sllr(DTR, LTR, l, DEVAL)
        min_dcf1 = compute_min_DCF(sllr1, sllr1, LEVAL, p, Cfn, Cfp)

        mindcf_logreg.append(min_dcf1)

        sllr2=weightedLogReg_sllr(DTR, LTR, l, DEVAL, p)
        min_dcf2 = compute_min_DCF(sllr2, sllr2, LEVAL, p, Cfn, Cfp)

        mindcf_weightedlogreg.append(min_dcf2)
    
        sllr3=qaudraticlogreg_sllr(DTR, LTR, l, DEVAL)
        min_dcf3 = compute_min_DCF(sllr3, sllr3, LEVAL, p, Cfn, Cfp)

        mindcf_quadraticlogreg.append(min_dcf3)
    
        print(l, min_dcf1, min_dcf2, min_dcf3)

    plt.xscale('log', base = 10)
    plt.plot(x, mindcf_logreg, label='minDCF logreg', color='b')
    plt.plot(x, mindcf_weightedlogreg, label='minDCF weighted logreg', color='r')
    plt.plot(x, mindcf_quadraticlogreg, label='minDCF quadratic logreg', color='g')
    plt.legend()
    plt.title("Logistic Regression variants evaluation set")
    plt.savefig("logreg variants evaluation set")
    plt.show()
    