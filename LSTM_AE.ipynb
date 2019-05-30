"""An LSTM (binary classifier) Autoencoder to identify rare events.
It is written using Tensorflow 2.0/Keras, Pandas and Seaborn libaries
Since I was playing arround with Autoencoders, I stumbled over this nice mini datascience project
and found it useful to reproduce it to learn to use Autoencoders with LSTM cells:
https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
The basic idea is:
- Train an LSTM autoencoder to generate/predict "normal" features for the next timesteop.
- If the measurment of the next timestep differs greatly from the generated/predicted, this is likely a fault sate
- use the few y = 1 data lines from the dataset just for validation and statistics and don't even bother to try to make a 

"""
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, CuDNNLSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve

#%%
#define constants
DATA_SPLIT_PCT = 0.2
LABELS = ["Normal","Break"]
LOOKBACK = 5
BATCH_SIZE = 64
EPOCHS = 1000


#%%
'''
The data is taken from https://arxiv.org/abs/1809.10717. Please use this source for any citation.
Download data here:
https://docs.google.com/forms/d/e/1FAIpQLSdyUk3lfDl7I5KYK_pw285LCApc-_RcoC0Tf9cnDnZ_TWzPAw/viewform
'''
df = pd.read_csv("processminer-rare-event-mts - data.csv") 
df_raw = df.copy()
df.tail(5) 

#%%
#look at this graph:
import seaborn as sns
df_small = df.sample(200)
print(df_small.shape)
sns.pairplot(df_small[df_small.keys()[2:10]],diag_kind="kde")

#%%
#shift data
def shift(df,shift_by):
    shift_by = abs(shift_by)
    df.y = df.y.shift(-shift_by)
    df = df[:-shift_by]
    df.y = df.y.astype('int32')
    return df

df = df_raw.copy()
shift_by = -2
df = shift(df,shift_by)
assert len(df_raw)-abs(shift_by) == len(df)
assert len(df[df.y.isna()]) == 0
df.tail(5)

#%%
# Remove time column, and the categorical columns
df = df.drop(['time', 'x28', 'x61'], axis=1)
df.keys()

#%%
#split into linex with y=0 and y=1. For the auto encoder we will only train y=0 samples.
df_x_y0 = df[df.y == 0]
df_x_y1 = df[df.y == 1]
#make numpy arrays x and y from the dataframes.
x_y0 = df_x_y0.loc[:, df_x_y0.columns != 'y'].values
#x_y1 = df_x_y1.loc[:, df_x_y1.columns != 'y'].values
y_y0 = df_x_y0['y'].values
#split into test and training data.
x_train, x_test, y_train, y_test = train_test_split(np.array(x_y0), np.array(y_y0), test_size=DATA_SPLIT_PCT)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=DATA_SPLIT_PCT)
print(f"x_train:{x_train.shape} x_test:{x_test.shape} x_valid:{x_valid.shape}")
#store the samples and freature count
n_samples = x_train.shape[0] 
n_features = x_train.shape[1] 
print(f"n_samples:{n_samples} n_features:{n_features}") 

#%%
#Normalize columns. Important to normalize test and train seperatly
def StdScale2DArray(df):
    values = df
    scaler = StandardScaler()
    values_scales = scaler.fit_transform(values)
    return values_scales

x_train_scaled = StdScale2DArray(x_train)
x_test_scaled = StdScale2DArray(x_test)
x_valid_scaled = StdScale2DArray(x_valid)
x_all_scaled = StdScale2DArray(x)
assert x_train_scaled.shape == x_train.shape


#%%
#LSTM requires 3d bachtes (batch x look_back x features)
#use Keras Timeseries generator to generate these: 
#TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
#For Auto Encoder, X and Y are identical.
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
generator = TimeseriesGenerator(x_train_scaled,x_train_scaled, length=LOOKBACK, batch_size=BATCH_SIZE)
generator_valid = TimeseriesGenerator(x_valid_scaled, x_valid_scaled, length=LOOKBACK, batch_size=BATCH_SIZE)
#a test batch to test and store shape
temp_x,_ = generator[0]
print(f"x: {temp_x.shape}")
timesteps =  temp_x.shape[1] # equal to the lookback
n_features =  temp_x.shape[2] # equal to features
print(f"timesteps: {timesteps}  n_features:{n_features}")


#%%
#Alternativly, if the fit_generator does not work, use this:
def AddTemporalDimension(df,lookback):
    new_df = np.empty((df.shape[0],lookback,df.shape[1]))
    for ii in range(lookback,df.shape[0]):
        new_df[ii] = df[ii-lookback:ii]
    return new_df

x_train_scaled_shaped = AddTemporalDimension(x_train_scaled,LOOKBACK)
x_valid_scaled_shaped = AddTemporalDimension(x_valid_scaled,LOOKBACK)
x_test_scaled_shaped = AddTemporalDimension(x_test_scaled,LOOKBACK)
timesteps =  x_train_scaled_shaped.shape[1] # equal to the lookback
n_features =  x_valid_scaled_shaped.shape[2] # equal to features
print(f"timesteps: {timesteps}  n_features:{n_features}")

#%%
#use this model declation if a) you are running pre-TF2.0 and b) have a cuda graphiccard configured.
#Otherwise use the declaration below
#Sequential model works just fine, no need for functional keras api
lstm_ae = Sequential()
#define encoder part
lstm_ae.add(CuDNNLSTM(timesteps, input_shape=(timesteps, n_features), return_sequences = True))
lstm_ae.add(CuDNNLSTM(16,return_sequences = True))
lstm_ae.add(CuDNNLSTM(1))
#make the encoder output match the inputs of decoder. 
#Without this, the input would be 2d and fail. Timesteps is always required for an LSTM
lstm_ae.add(RepeatVector(timesteps))
#define decoder part. 
lstm_ae.add(CuDNNLSTM(timesteps,return_sequences = True))
lstm_ae.add(CuDNNLSTM(16,return_sequences = True))
lstm_ae.add(TimeDistributed(Dense(n_features)))
#compile
lstm_ae.compile(loss='mse', optimizer='adam')
lstm_ae.summary()

#%%
#Alternativly: use this verstion if you are on TF2.0.
lstm_ae = Sequential()
#define encoder part
lstm_ae.add(LSTM(timesteps, input_shape=(timesteps, n_features), activation = 'relu', return_sequences = True))
lstm_ae.add(LSTM(16, activation = 'relu',return_sequences = True))
lstm_ae.add(LSTM(1, activation = 'relu'))
#make the encoder output match the inputs of decoder. 
#Without this, the input would be 2d and fail. Timesteps is always required for an LSTM
lstm_ae.add(RepeatVector(timesteps))
#define decoder part. 
lstm_ae.add(LSTM(timesteps, activation = 'relu',return_sequences = True))
lstm_ae.add(LSTM(16, activation = 'relu',return_sequences = True))
lstm_ae.add(TimeDistributed(Dense(n_features)))
#compile
lstm_ae.compile(loss='mse', optimizer=optimizers.Adam(0.0001))
lstm_ae.summary()

#%%
#add checkpoints and fit
cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                               save_best_only=True,
                               verbose=0)

tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

EPOCHS = 1000


lstm_autoencoder_history = lstm_ae.fit(x_train_scaled_shaped, x_train_scaled_shaped, 
                                                epochs=EPOCHS, 
                                                batch_size=BATCH_SIZE,
                                                validation_data=(x_valid_scaled_shaped,x_valid_scaled_shaped),
                                                steps_per_epoch=10,
                                                verbose=2).history      

#Alternativly: Use the Generator method. Cooler, saves memory, but Keras has a reptorted bug on when I wrote this
#lstm_autoencoder_history = lstm_ae.fit_generator(generator, epochs=EPOCHS, steps_per_epoch=10,verbose=2).history                                          

#batch_size=BATCH_SIZE,


#%%
plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#%%
def flatten(X):
    """Undo the the AddTemporalDimension.
    """
    flattened_X = np.empty((X.shape[0], X.shape[2]))  
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

#%%
test_x_predictions = lstm_ae.predict(x_test_scaled_shaped)
mse = np.mean(np.power(flatten(x_test_scaled_shaped) - flatten(test_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test.tolist()})

threshold_fixed = 0.8
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()


#%%
#make the test dataset. Just use all data here
x_all = df.loc[:, df.columns != 'y'].values
x_all_scaled = StdScale2DArray(x_all)
y_all = df.y.values
assert len(x_all_scaled) == len(y_all)


#%%
x_to_test = AddTemporalDimension(x_all_scaled,LOOKBACK)
y_to_test = y_all
threshold_fixed = 0.8 #the threshold from where on it is counted as an anormaly = machine failure

test_x_predictions = lstm_ae.predict(x_to_test)
Reconstruction_error = np.mean(np.power(flatten(x_to_test) - flatten(test_x_predictions), 2), axis=1)
pred_y = [1 if e > threshold_fixed else 0 for e in Reconstruction_error] #make prediction of y binary
conf_matrix = confusion_matrix(y_to_test.tolist(), pred_y)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


#%%
false_pos_rate, true_pos_rate, thresholds = roc_curve(y_to_test, Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


