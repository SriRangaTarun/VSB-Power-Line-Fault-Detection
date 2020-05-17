import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.layers import *
from keras.models import *
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

N_SPLITS = 5
max_num = 127
min_num = -128
sample_size = 800000

parser = argparse.ArgumentParser()
parser.add_argument('test_data_path')
parser.add_argument('train_data_path')
parser.add_argument('test_parquet_path')
parser.add_argument('train_parquet_path')

args = parser.parse_args()
test_data_path = args.test_data_path
train_data_path = args.train_data_path
test_parquet_path = args.test_parquet_path
train_parquet_path = args.train_parquet_path

df_train = pd.read_csv(train_data_path)
df_train = df_train.set_index(['id_measurement', 'phase'])

X, y = [], []
def load_all():
    total_size = len(df_train)
    for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
        X_temp, y_temp = prep_data(ini, end, train_parquet)
        X.append(X_temp)
        y.append(y_temp)
load_all()
X = np.concatenate(X)
y = np.concatenate(y)

def entropy_and_fractal_dim(x):
    return [perm_entropy(x), svd_entropy(x), app_entropy(x),\
            sample_entropy(x), petrosian_fd(x), katz_fd(x), higuchi_fd(x)]

features = []
signals = X.reshape((len(X), X.shape[1]*X.shape[2]))
for signal in signals: features.append(entropy_and_fractal_dim(signal))

scaler = MinMaxScaler(feature_range=(0, 1))
features = np.array(features).reshape((len(features), 7))
scaler.fit(features); features = scaler.transform(features)

def model_lstm(input_shape, feat_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    feat = Input(shape=(feat_shape[1],))

    bi_lstm = Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(inp)
    bi_gru = Bidirectional(CuDNNGRU(64, return_sequences=True), merge_mode='concat')(bi_lstm)
    
    attention = Attention(input_shape[1])(bi_gru)
    
    x = concatenate([attention, feat], axis=1)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[inp, feat], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    
    return model

y_val = []
preds_val = []
splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=10).split(X, y))

for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session()
    print("Beginning fold {}".format(idx+1))
    data = X[train_idx], features[train_idx], y[train_idx], X[val_idx], features[val_idx], y[val_idx]

    model = model_lstm(train_X.shape, features.shape)
    train_X, train_feat, train_y, val_X, val_feat, val_y = data
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True,
                           save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
   
    model.fit([train_X, train_feat], train_y, batch_size=128,
              epochs=50, validation_data=([val_X, val_feat], val_y), callbacks=[ckpt])
    
    y_val.append(val_y)
    model.load_weights('weights_{}.h5'.format(idx))
    preds_val.append(model.predict([val_X, val_feat], batch_size=512))

preds_val = np.concatenate(preds_val)[...,0]; y_val = np.concatenate(y_val)

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = K.eval(matthews_correlation(K.variable(y_true.astype("float64")),
                                            K.variable((y_proba > threshold).astype("float64"))))
        if score > best_score:
            best_threshold = threshold; best_score = score

    return {'threshold': best_threshold, 'matthews_correlation': best_score}

optimal_values = threshold_search(y_val, preds_val)
best_threshold, best_score = optimal_values['threshold'], optimal_values['matthews_correlation']

print("Optimal Threshold : " + str(best_threshold))
print("Best Matthews Correlation : " + str(best_score))

meta_test = pd.read_csv(test_data_path)

n_parts = 10
max_line = len(meta_test)
first_sig = meta_test.index[0]
last_part = max_line % n_parts
part_size = int(max_line / n_parts)

start_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]
start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]

X_test = []
for start, end in start_end:
    cols = [str(i) for i in range(start, end)]
    subset_test = pq.read_pandas(test_parquet_path, columns=cols)
    
    subset_test = subset_test.to_pandas()
    for i in tqdm(subset_test.columns):
        subset_test_col = subset_test[i]
        subset_trans = transform_ts(subset_test_col)
        id_measurement, phase = meta_test.loc[int(i)]
        X_test.append([i, id_measurement, phase, subset_trans])
        
X_test_input = np.asarray([np.concatenate([X_test[i][3],
                                           X_test[i+1][3],
                                           X_test[i+2][3]], axis=1) for i in range(0,len(X_test), 3)])

signals = X_test_input.reshape((len(X_test_input), X_test_input.shape[1]*X_test_input.shape[2]))

features_test = []
for signal in signals:
    features_test.append(entropy_and_fractal_dim(signal))
    
features_test = scaler.transform(np.array(features_test).reshape((len(features_test), 7)))

preds_test = []
submission = pd.read_csv('../input/sample_submission.csv')

for i in range(N_SPLITS):
    model.load_weights('weights_{}.h5'.format(i))
    pred = model.predict([X_test_input, features_test],
                         batch_size=300, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)

preds = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)

submission['target'] = preds
submission.to_csv('submission.csv', index=False)
