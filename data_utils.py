def matthews_correlation(y_true, y_pred):

    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_pos = K.round(y_pred_pos)

    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    ts_std = (ts  - min_data) / (max_data - min_data)
    return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]
    
def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    bucket_size = int(sample_size / n_dim)
    ts_std = min_max_transf(ts, min_num, max_num)

    new_ts = []
    for i in range(0, sample_size, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
 
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 

        relative_percentile = percentil_calc - mean
        max_range = percentil_calc[-1] - percentil_calc[0]
        data_points = np.asarray([mean, std, std_top, std_bot, max_range])
        new_ts.append(np.concatenate([data_points, percentil_calc, relative_percentile]))

    return np.asarray(new_ts)

def prep_data(start, end):
    X, y = [], []
    for id_measurement in tqdm(df_train.index.levels[0].unique()[int(start/3):int(end/3)]):

        X_signal = []
        for phase in [0,1,2]:
            signal_id, target = df_train.loc[id_measurement].loc[phase]
 
            if phase == 0:
                y.append(target)
            X_signal.append(transform_ts(praq_train[str(signal_id)]))

        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y
