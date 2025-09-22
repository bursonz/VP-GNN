import numpy as np
import torch


def mask_normalize(Pf: np.ndarray, mf: np.ndarray, stdf: np.ndarray):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    mf = mf.reshape((-1, 1))
    stdf = stdf.reshape((-1, 1))
    
    N, T, F = Pf.shape
    mask = 1*(Pf > 0) + 0*(Pf <= 0)
    Pf = Pf.transpose((2, 0, 1)).reshape(F, -1)
    mask = mask.transpose((2, 0, 1)).reshape(F, -1)
    Pf = (Pf - mf) / (stdf + 1e-18)
    Pf = Pf * mask

    Pnorm = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    mask = mask.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal = np.concatenate([Pnorm, mask], axis=2)
    return Pfinal


def mask_normalize_static(Ps: np.ndarray, ms: np.ndarray, ss: np.ndarray):
    ms = ms.reshape((-1, 1))
    ss = ss.reshape((-1, 1))

    N, S = Ps.shape
    Ps = Ps.transpose((1, 0))

    # input normalization
    Ps = (Ps - ms) / (ss + 1e-18)
    # for s in range(S):
    #     Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm = Ps.transpose((1, 0))
    return Pnorm



def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    nP = len(P)
    Ptemp = [P[i]['arr'] for i in range(nP)]
    Ptemp = np.stack(Ptemp)
    Ptemp = mask_normalize(Ptemp, mf, stdf)
    Ptemp = torch.from_numpy(Ptemp)

    Ptime = [P[i]['time'] for i in range(nP)]
    Ptime = np.stack(Ptime)
    Ptime = torch.from_numpy(Ptime) / 60.0  # convert mins to hours
    
    Pstatic = [P[i]['extended_static'] for i in range(nP)]
    Pstatic = mask_normalize_static(Pstatic, ms, ss)
    Pstatic = torch.from_numpy(Pstatic)

    Ys = torch.from_numpy(y[:, 0]).type(torch.LongTensor)
    return Ptemp, Pstatic, Ptime, Ys


def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape
    Ptime = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        Ptime[i] = tim

    Pf = mask_normalize(P, mf, stdf)
    Pf = torch.Tensor(Pf)

    Ptime = torch.Tensor(Ptime) / 60.0

    Ys = torch.from_numpy(y[:, 0]).type(torch.LongTensor)
    return Pf, None, Ptime, Ys


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F,))
    stdf = np.ones((F,))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f].item(), eps])
    return mf, stdf


def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S,))
    ss = np.ones((S,))

    if dataset == 'P12':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == 'eICU':
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical[s] == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss


def tensorize_normalize_and_remove_part(dataset, Ptrain, Pval, Ptest, Ytrain, Yval, Ytest, feature_removal_level, missing_ratio):
    if dataset in {'P12', 'P19', 'eICU'}:
        TrainF = [Ptrain[i]['arr'] for i in range(len(Ptrain))]
        TrainT = [Ptrain[i]['time'] for i in range(len(Ptrain))]
        TrainS = [Ptrain[i]['extended_static'] for i in range(len(Ptrain))]
        TrainF = np.stack(TrainF) # #samples, #time, #variables
        TrainT = np.stack(TrainT) # #samples, #time, 1
        TrainS = np.stack(TrainS) # #samples, #variables

        mf, stdf = getStats(TrainF)
        ms, ss = getStats_static(TrainS, dataset=dataset)

        TrainF = mask_normalize(TrainF, mf, stdf)
        TrainS = mask_normalize_static(TrainS, ms, ss)
        TrainF = torch.from_numpy(TrainF).to(torch.float32)
        TrainT = torch.from_numpy(TrainT).to(torch.float32) / 60.0  # convert mins to hours
        TrainS = torch.from_numpy(TrainS).to(torch.float32)
        TrainY = torch.from_numpy(Ytrain[:,0]).to(torch.long)

        ValF = [Pval[i]['arr'] for i in range(len(Pval))]
        ValT = [Pval[i]['time'] for i in range(len(Pval))]
        ValS = [Pval[i]['extended_static'] for i in range(len(Pval))]
        ValF = np.stack(ValF) # #samples, #time, #variables
        ValT = np.stack(ValT) # #samples, #time, 1
        ValS = np.stack(ValS) # #samples, #variables
        ValF = mask_normalize(ValF, mf, stdf)
        ValS = mask_normalize_static(ValS, ms, ss)
        ValF = torch.from_numpy(ValF).to(torch.float32)
        ValT = torch.from_numpy(ValT).to(torch.float32) / 60.0  # convert mins to hours
        ValS = torch.from_numpy(ValS).to(torch.float32)
        ValY = torch.from_numpy(Yval[:,0]).to(torch.long)

        TestF = [Ptest[i]['arr'] for i in range(len(Ptest))]
        TestT = [Ptest[i]['time'] for i in range(len(Ptest))]
        TestS = [Ptest[i]['extended_static'] for i in range(len(Ptest))]
        TestF = np.stack(TestF) # #samples, #time, #variables
        TestT = np.stack(TestT) # #samples, #time, 1
        TestS = np.stack(TestS) # #samples, #variables
        TestF = mask_normalize(TestF, mf, stdf)
        TestS = mask_normalize_static(TestS, ms, ss)
        TestF = torch.from_numpy(TestF).to(torch.float32)
        TestT = torch.from_numpy(TestT).to(torch.float32) / 60.0  # convert mins to hours
        TestS = torch.from_numpy(TestS).to(torch.float32)
        TestY = torch.from_numpy(Ytest[:,0]).to(torch.long)

    elif dataset == 'PAM':
        pass
        # D = 1
        # TrainF = Ptrain
        # TrainS = np.zeros((len(Ptrain), D))

        # mf, stdf = getStats(Ptrain)
        # TrainF, TrainS, TrainT, TrainY = tensorize_normalize_other(Ptrain, Ytrain, mf, stdf)
        # ValF, ValS, ValT, ValY = tensorize_normalize_other(Pval, Yval, mf, stdf)
        # TestF, TestS, TestT, TestY = tensorize_normalize_other(Ptest, Ytest, mf, stdf)

    # remove part of variables in validation and test set
    if missing_ratio > 0:
        num_all_features =int(ValF.shape[2] / 2)
        num_missing_features = round(missing_ratio * num_all_features)
        if feature_removal_level == 'sample':
            for i, patient in enumerate(ValF):
                idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                patient[:, idx] = torch.zeros(ValF.shape[1], num_missing_features)
                ValF[i] = patient
            for i, patient in enumerate(TestF):
                idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                patient[:, idx] = torch.zeros(TestF.shape[1], num_missing_features)
                TestF[i] = patient
        elif feature_removal_level == 'set':
            density_score_indices = np.load('./baselines/saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
            idx = density_score_indices[:num_missing_features].astype(int)
            ValF[:, :, idx] = torch.zeros(ValF.shape[0], ValF.shape[1], num_missing_features)
            TestF[:, :, idx] = torch.zeros(TestF.shape[0], TestF.shape[1], num_missing_features)

    # TrainF = TrainF.permute(1, 0, 2)
    # ValF = ValF.permute(1, 0, 2)
    # TestF = TestF.permute(1, 0, 2)

    # TrainT = TrainT.squeeze(2).permute(1, 0)
    # ValT = ValT.squeeze(2).permute(1, 0)
    # TestT = TestT.squeeze(2).permute(1, 0)
    return TrainF, TrainS, TrainT, TrainY, ValF, ValS, ValT, ValY, TestF, TestS, TestT, TestY
