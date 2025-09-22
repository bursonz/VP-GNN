import numpy as np
import logging
import pickle as pkl
import os


def get_data_split(dataset, logger, split_type='random', seed=0, split=0, reverse=False, baseline=True, predictive_label='mortality'):
    base_path = f'datasets/{dataset.lower()}'
    processed_path = f'{base_path}/processed_data'
    split_path = f'{base_path}/splits/seed{seed}_split{split}.pkl'

    # load data
    if dataset == 'P12':
        Pdict_arr = np.load(f'{processed_path}/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(f'{processed_path}/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_arr = np.load(f'{processed_path}/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(f'{processed_path}/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'eICU':
        Pdict_arr = np.load(f'{processed_path}/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(f'{processed_path}/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = 'eICU_'
    elif dataset == 'PAM':
        Pdict_arr = np.load(f'{processed_path}/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(f'{processed_path}/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

    if split_type == 'random':
        with open(split_path, 'rb') as rbfile:
            idx_train, idx_val, idx_test = pkl.load(rbfile)

    # extract train/val/test examples
    Ptrain = Pdict_arr[idx_train]
    Pval = Pdict_arr[idx_val]
    Ptest = Pdict_arr[idx_test]

    if dataset == 'P12' or dataset == 'PAM':
        if predictive_label == 'mortality':
            y = arr_outcomes[:, -1].reshape((-1, 1))
        elif predictive_label == 'LoS':  # for P12 only
            y = arr_outcomes[:, 3].reshape((-1, 1))
            y = np.array(list(map(lambda los: 0 if los <= 3 else 1, y)))[..., np.newaxis]
    elif dataset == 'P19':
        y = arr_outcomes.reshape((-1, 1))
    elif dataset == 'eICU':
        y = arr_outcomes[..., np.newaxis]
    Ytrain = y[idx_train]
    Yval = y[idx_val]
    Ytest = y[idx_test]

    logger.info(f'Train dataset: {len(Ptrain)}, {len(Ytrain)}')
    logger.info(f' Val  dataset: {len(Pval)}, {len(Yval)}')
    logger.info(f'Test  dataset: {len(Ptest)}, {len(Ytest)}')
    return Ptrain, Pval, Ptest, Ytrain, Yval, Ytest


def get_features_mean(X_features):
    """
    Calculate means of all time series features (36 features in P12 dataset).

    :param X_features: time series features for all samples in training set
    :return: list of means for all features
    """
    samples, timesteps, features = X_features.shape
    X = np.reshape(X_features, newshape=(samples*timesteps, features)).T
    means = []
    for row in X:
        row = row[row > 0]
        means.append(np.mean(row))
    return means


def mean_imputation(X_features, X_time, mean_features, missing_value_num):
    """
    Fill X_features missing values with mean values of all train samples.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :param mean_features: mean values of features from the training set
    :return: X_features, filled with mean values instead of zeros (missing observations)
    """
    time_length = []
    for times in X_time:
        if np.where(times == missing_value_num)[0].size == 0:
            time_length.append(times.shape[0])
        elif np.where(times == missing_value_num)[0][0] == 0:
            time_length.append(np.where(times == missing_value_num)[0][1])
        else:
            time_length.append(np.where(times == missing_value_num)[0][0])

    # check for inconsistency
    for i in range(len(X_features)):
        if np.any(X_features[i, time_length[i]:, :]):
            print('Inconsistency between X_features and X_time: features are measured without time stamp.')

    # impute times series features
    for i, sample in enumerate(X_features):
        X_features_relevant = sample[:time_length[i], :]
        missing_values_idx = np.where(X_features_relevant == missing_value_num)
        for row, col in zip(*missing_values_idx):
            X_features[i, row, col] = mean_features[col]

    return X_features


def forward_imputation(X_features, X_time, missing_value_num):
    """
    Fill X_features missing values with values, which are the same as its last measurement.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with last measurements instead of zeros (missing observations)
    """
    time_length = []
    for times in X_time:
        if np.where(times == missing_value_num)[0].size == 0:
            time_length.append(times.shape[0])
        elif np.where(times == missing_value_num)[0][0] == 0:
            time_length.append(np.where(times == missing_value_num)[0][1])
        else:
            time_length.append(np.where(times == missing_value_num)[0][0])

    # impute times series features
    for i, sample in enumerate(X_features):
        for j, ts in enumerate(sample.T):   # note the transposed matrix
            first_observation = True
            current_value = -1
            for k, observation in enumerate(ts[:time_length[i]]):
                if X_features[i, k, j] == missing_value_num and first_observation:
                    continue
                elif X_features[i, k, j] != missing_value_num:
                    current_value = X_features[i, k, j]
                    first_observation = False
                elif X_features[i, k, j] == missing_value_num and not first_observation:
                    X_features[i, k, j] = current_value

    return X_features


def cubic_spline_imputation(X_features, X_time, missing_value_num):
    """
    Fill X_features missing values with cubic spline interpolation.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with interpolated values
    """
    from scipy.interpolate import CubicSpline

    time_length = []
    for times in X_time:
        if np.where(times == missing_value_num)[0].size == 0:
            time_length.append(times.shape[0])
        elif np.where(times == missing_value_num)[0][0] == 0:
            time_length.append(np.where(times == missing_value_num)[0][1])
        else:
            time_length.append(np.where(times == missing_value_num)[0][0])

    # impute times series features
    for i, sample in enumerate(X_features):
        for j, ts in enumerate(sample.T):   # note the transposed matrix
            valid_ts = ts[:time_length[i]]
            zero_idx = np.where(valid_ts == missing_value_num)[0]
            non_zero_idx = np.nonzero(valid_ts)[0]
            y = valid_ts[non_zero_idx]

            if len(y) > 1:   # we need at least 2 observations to fit cubic spline
                x = X_time[i, :time_length[i], 0][non_zero_idx]
                x2interpolate = X_time[i, :time_length[i], 0][zero_idx]

                cs = CubicSpline(x, y)
                interpolated_ts = cs(x2interpolate)
                valid_ts[zero_idx] = interpolated_ts

                # set values before first measurement to the value of first measurement
                first_obs_index = non_zero_idx[0]
                valid_ts[:first_obs_index] = np.full(shape=first_obs_index, fill_value=valid_ts[first_obs_index])

                # set values after last measurement to the value of last measurement
                last_obs_index = non_zero_idx[-1]
                valid_ts[last_obs_index:] = np.full(shape=time_length[i] - last_obs_index, fill_value=valid_ts[last_obs_index])

                X_features[i, :time_length[i], j] = valid_ts

    return X_features


def imputation_(method, dataset, Ptrain, Pval, Ptest):

    # impute missing values
    if method != 'no_imputation':
        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            X_features_train = np.array([d['arr'] for d in Ptrain])
            X_time_train = np.array([d['time'] for d in Ptrain])
            X_features_val = np.array([d['arr'] for d in Pval])
            X_time_val = np.array([d['time'] for d in Pval])
            X_features_test = np.array([d['arr'] for d in Ptest])
            X_time_test = np.array([d['time'] for d in Ptest])
        elif dataset == 'PAM':
            X_features_train = Ptrain
            X_time_train = np.array([np.arange(1, Ptrain.shape[1] + 1)[..., np.newaxis] for d in Ptrain])
            X_features_val = Pval
            X_time_val = np.array([np.arange(1, Pval.shape[1] + 1)[..., np.newaxis] for d in Pval])
            X_features_test = Ptest
            X_time_test = np.array([np.arange(1, Ptest.shape[1] + 1)[..., np.newaxis] for d in Ptest])

        if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM':
            missing_value_num = 0
        elif dataset == 'eICU':
            missing_value_num = -1

        if method == 'mean':
            features_means = get_features_mean(X_features_train)
            X_features_train = mean_imputation(X_features_train, X_time_train, features_means, missing_value_num)
            X_features_val = mean_imputation(X_features_val, X_time_val, features_means, missing_value_num)
            X_features_test = mean_imputation(X_features_test, X_time_test, features_means, missing_value_num)
        elif method == 'forward':
            X_features_train = forward_imputation(X_features_train, X_time_train, missing_value_num)
            X_features_val = forward_imputation(X_features_val, X_time_val, missing_value_num)
            X_features_test = forward_imputation(X_features_test, X_time_test, missing_value_num)
        elif method == 'cubic_spline':
            X_features_train = cubic_spline_imputation(X_features_train, X_time_train, missing_value_num)
            X_features_val = cubic_spline_imputation(X_features_val, X_time_val, missing_value_num)
            X_features_test = cubic_spline_imputation(X_features_test, X_time_test, missing_value_num)

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            for i, pat in enumerate(X_features_train):
                Ptrain[i]['arr'] = pat
            for i, pat in enumerate(X_features_val):
                Pval[i]['arr'] = pat
            for i, pat in enumerate(X_features_test):
                Ptest[i]['arr'] = pat
        elif dataset == 'PAM':
            for i, pat in enumerate(X_features_train):
                Ptrain[i] = pat
            for i, pat in enumerate(X_features_val):
                Pval[i] = pat
            for i, pat in enumerate(X_features_test):
                Ptest[i] = pat


def setup_logging(dataset, model_name, seed, missing_ratio):
    logger = logging.getLogger('main')
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt="%Y/%m/%d %H:%M:%S")

    folder = f'logs/{dataset}/{missing_ratio}/{model_name}'
    os.makedirs(folder, exist_ok=True)
    handler = logging.FileHandler(f'{folder}/seed{seed}.log', 'a')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def random_sample(idx_0, idx_1, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(idx_0, size=int(B / 2), replace=replace)
    idx1_batch = np.random.choice(idx_1, size=int(B / 2), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx
