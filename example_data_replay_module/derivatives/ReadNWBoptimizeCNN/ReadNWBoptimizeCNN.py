import json
import logging
import numbers
import numpy as np
import os
import redis
from redis import ConnectionError, Redis
import signal
import sys
import yaml
import pdb
import gc
import time
import pandas as pd

# Node specific imports
from tensorflow import keras
from keras.layers import (Input, Flatten, Dense, Activation, Permute, Dropout, Conv1D, Conv2D, MaxPooling2D,
                          AveragePooling2D, SeparableConv2D, DepthwiseConv2D,BatchNormalization)
from keras.constraints import max_norm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import adam_v2
from keras import backend as K
import tensorflow.lite

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

import matplotlib.pyplot as plt


## set up logging (same for all derivatives)
NAME = 'train_CNN'
loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{NAME}] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)
logging.info('Derivative started')

# set up clean exit code (same for all derivatives)
def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(0)

# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)

parameters_command_line = dict()
parameters_command_line['full_rdb_save_path'] = sys.argv[1]
parameters_command_line['host_ip'] = sys.argv[2]
parameters_command_line['host_port'] = sys.argv[3]
parameters = sys.argv[4]

# Load parameters detailed in the graph
parameters = json.loads(parameters)

# Data formatting parameters
Data_parameters = parameters['Data_parameters']
Datafile = parameters['output_nwb_filename']
with open(Data_parameters['stream_definitions'], 'r') as file:
    stream_definitions = yaml.safe_load(file)
X_variable = Data_parameters['X_variable']
X_info = stream_definitions['RedisStreams']['Outputs'][X_variable]['samples']
Y_variable = Data_parameters['Y_variable']
Y_info = stream_definitions['RedisStreams']['Outputs'][Y_variable]['samples']
Y_labels = np.array(Y_info['y_labels'])
fs = X_info['frequency']
refresh_freq = X_info['refresh_freq']
window_size = fs // refresh_freq
channels = X_info['chan_per_stream']

# Training parameters
Training_parameters = parameters['Training_parameters']
epoch_count = Training_parameters['epoch_count']
learning_rate = Training_parameters['learning_rate']
standardize_input = Training_parameters['standardize_input']
standardize_output = Training_parameters['standardize_output']
val_split = Training_parameters['val_split']
model_filename = Training_parameters['model_filename']
earlystopping_patience = Training_parameters['earlystopping_patience']
batch_size = Training_parameters['batch_size']
test_split = Training_parameters['test_split']
sliding_window = int(0.5*window_size)
logging.info(str(model_filename))

# Not yet intialized by system
monitor_loss = 'val_loss'

# If testing, this will point to some hard coded folder, but will instead grab the current directory when used with BRAND
if testing == True:
    base_folder = '/home/alex/Desktop/Data/EMG'
else:
    base_folder = os.path.dirname(os.path.abspath(__file__))

logging.info(str(base_folder))
performances_save_folder = base_folder + '/performances/'
checkpointPath = base_folder + '/checkpoints/check' + time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime()) + ".hdf5"
save_file = base_folder + '/performances/Continuous_predictions' + time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime()) + '.csv'
verbosity = 1
save_performances = 0
save_kins = 0

# Optimization parameters
Optimization_parameters = parameters['Optimization_parameters']
dropout_rate_list = Optimization_parameters['dropout_rate_list']
max_conv_blocks_list = Optimization_parameters['max_conv_blocks_list']
con_type_list = Optimization_parameters['con_type_list']
mlp_units_list = Optimization_parameters['mlp_units_list']
hidden_units_list = Optimization_parameters['hidden_units_list']
output_type = Optimization_parameters['output_type']
filter_size_list = Optimization_parameters['filter_size_list']

# Create the parameter list based on the inputs ranges
param_list = list(ParameterGrid({'1_model_type': ['CNN'], 'dropoutRate': dropout_rate_list, 'max_conv_blocks': max_conv_blocks_list,
     'con_type': con_type_list, 'num_mlp_units': mlp_units_list, 'hidden_units': hidden_units_list,
     'output_type': output_type, 'filter_size': filter_size_list}))


# This general CNN allows you to modify the CNN architecture for a number of input parameters
def main_CNN_general(nb_classes=3, Chans=64, Samples=256, dropoutRate=0.5, max_conv_blocks=2, con_type='dense',
                     num_mlp_units=[256], hidden_units=32, output_type='discrete', filter_size = 10):

    # Define CNN modelhere

    return keras.models.Model(inputs=input_main, outputs=outputs)

# Performance labels (has not been tested, but will need to be expanded upon for actual experiments)
performance_labels = ['Train MSE', 'Validation MSE', 'Test MSE', 'Window Size', 'Last Epoch', 'Param_grid', 'time_trained']
performance_data = pd.DataFrame([], columns=performance_labels)

# Open the nwb file created with the train graph
with NWBHDF5IO(str(Datafile), 'r') as io:
    nwbfile = io.read()
    X_data = np.swapaxes(nwbfile.acquisition[X_variable].data[:], 0, 1)
    Y_data = np.swapaxes(nwbfile.acquisition[Y_variable].spatial_series[Y_variable].data[:], 0, 1)
    if len(Y_data.shape) == 1:
        Y_data = np.expand_dims(Y_data,1)
    timestamps = nwbfile.acquisition[X_variable].timestamps[:]

# Data initialization (MUCH FASTER)
X = np.zeros((int(np.floor(np.shape(Y_data)[1])/sliding_window), channels, window_size))
Y = np.zeros((int(np.floor(np.shape(Y_data)[1])/sliding_window), len(Y_labels)))

# Epoch data based on sliding windows
for i in range(0, np.shape(X)[0]-int(window_size/sliding_window)):
    X[i, :, :] = X_data[:, int((i)*sliding_window):int((i)*sliding_window+window_size)]
    Y[i, :] = Y_data[:, int(i*sliding_window+window_size)]
del X_data, Y_data

# Segment Val and Test sets (with NO shuffle as that is not realistic for (most) neuro experiments)
# Cut off the test set first so you are testing the model on the final epochs in the temporal sense
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split, shuffle=False)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_split, shuffle=False)

# Standardize the input in a per-channel manner
if standardize_input == 1:
    scalers = {}
    for i in range(X_train.shape[2]):
        scalers[i] = StandardScaler()
        X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])

    for i in range(X_test.shape[2]):
        X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])

    for i in range(X_val.shape[2]):
        X_val[:, :, i] = scalers[i].transform(X_val[:, :, i])

# Standardize the output in a per-channel manner
if standardize_output == 1:
    scalers_output = {}
    for i in range(Y_train.shape[1]):
        scalers_output[i] = StandardScaler()
        Y_train[:, i] = np.squeeze(scalers_output[i].fit_transform(Y_train[:, i].reshape(-1, 1)))

    for i in range(Y_test.shape[1]):
        Y_test[:, i] = np.squeeze(scalers_output[i].transform(Y_test[:, i].reshape(-1, 1)))

    for i in range(Y_val.shape[1]):
        Y_val[:, i] = np.squeeze(scalers_output[i].transform(Y_val[:, i].reshape(-1, 1)))

# Format for CNN (as you need a dimension for CNN channels (like RGB channels for pictures))
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)
X_val = np.expand_dims(X_val, 3)

# Initialize the metric (lower is better for  MSE)
metric_comparison = 999999
for i in range(len(param_list)):

    # Select the current parameter group
    parameters = param_list[i]
    logging.info(str(parameters))
    logging.info(str('Model # ' + str(i) + ' of ' + str(len(param_list)) + ' total models.'))

    # Grab the new model with the current set of parameters
    model = main_CNN_general(nb_classes=len(Y_labels), Chans=channels, Samples=window_size,
                             dropoutRate=parameters['dropoutRate'],
                             max_conv_blocks=parameters['max_conv_blocks'], con_type=parameters['con_type'],
                             num_mlp_units=parameters['num_mlp_units'], hidden_units=parameters['hidden_units'],
                             output_type=parameters['output_type'], filter_size=parameters['filter_size'])

    opt = adam_v2.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=[keras.metrics.MeanSquaredError()])

    # Print out a text summary of the architecture
    model.summary()

    # Define the ModelCheckpoint and earlystopping callback
    checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=verbosity, save_best_only=True, monitor=monitor_loss)
    earlystopping = EarlyStopping(monitor=monitor_loss, min_delta=0, patience=earlystopping_patience)

    # Fit model
    fittedModel = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch_count,
                            verbose=verbosity, validation_data=(X_val, Y_val),
                            callbacks=[checkpointer, earlystopping], shuffle='yes')

    # Load the model with the best validation loss, and evaluate
    model.load_weights(checkpointPath)

    # Get scores with loaded model
    score = model.evaluate(X_test, Y_test, verbose=0)
    score_train = model.evaluate(X_train, Y_train, verbose=0)
    score_val = model.evaluate(X_val, Y_val, verbose=0)

    # Get predictions themselves
    probs = model.predict(X_test)
    probs_train = model.predict(X_train)
    probs_val = model.predict(X_val)

    # Log MSE scores and the number of unique values 
    # If the model is breaking in some way, you may see a very low number of unique values
    # Indicating the model is predicting the same thing for different inputs
    logging.info(str('Train MSE:' + str(score_train[1]) + ' with '+ str( len(np.unique(probs_train)))+ ' unique values (original unique values: '+ str( len(np.unique(Y_train)))+ ')'))
    logging.info(str('Validation MSE:'+ str(score_val[1])+ ' with '+ str(len(np.unique(probs_val)))+  ' unique values (original unique values: '+ str( len(np.unique(Y_val)))+  ')'))
    logging.info(str('Test MSE:'+ str(score[1])+  ' with '+ str( len(np.unique(probs)))+  ' unique values (original unique values: '+ str( len(np.unique(Y_test)))+  ')'))

    # R2 may be a better way to understand decoding performance
    r2_weighted = r2_score(Y_test, probs, multioutput='variance_weighted')
    r2_train_weighted = r2_score(Y_train, probs_train, multioutput='variance_weighted')
    r2_val_weighted = r2_score(Y_val, probs_val, multioutput='variance_weighted')
    
    # Log R2 Scores
    logging.info(str('weighted train r2'+ str( r2_train_weighted)))
    logging.info(str('weighted val r2'+ str(r2_val_weighted)))
    logging.info(str('weighted test r2'+ str( r2_weighted)))

    # Grab the final epoch to see what 'early stopping' is doing
    final_epoch = str(len(fittedModel.history['loss']))

    # Initialize the model to be saved
    if i == 1:
        model_to_save = model

    # Check if the current model outperforms (lower score) the previous models
    if score[1] <= metric_comparison:
        model_to_save = model
        metric_comparison = score[1]

    # Delete the model when you grab the one to save as overwriting the model can sometimes cause problems with GPU memory
    del model, fittedModel

    # de-standardize the outputs (so you can save if you want)
    if standardize_output == 1:
        for i in range(Y_test.shape[1]):
            Y_test[:, i] = np.squeeze(scalers_output[i].inverse_transform(Y_test[:, i].reshape(-1, 1)))
            probs[:, i] = np.squeeze(scalers_output[i].inverse_transform(probs[:, i].reshape(-1, 1)))

    # For plotting, etc
    time_vector = np.expand_dims(np.linspace(0, len(probs) / fs, len(probs)), 1)

    current_time = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())

    if save_kins == 1:

        time_vector = np.expand_dims(np.linspace(0, len(probs)/fs, len(probs)), 1)
        probs_with_time = np.concatenate((time_vector, probs), axis=1)
        pd_probs_with_time = pd.DataFrame(probs_with_time, columns = ['time'] + Y_labels[Y_labels])
        pd_probs_with_time.to_csv(performances_save_folder + "kins/kins_pred_with_time_" + current_time + ".csv", index=False)

        real_with_time = np.concatenate((time_vector, Y_test), axis=1)
        pd_real_with_time = pd.DataFrame(real_with_time, columns = ['time'] + Y_labels[Y_labels])
        pd_real_with_time.to_csv(performances_save_folder + "kins/kins_true_with_time.csv", index=False)

    # You'll want to save performances for every model when actually optimizing
    if save_performances==1:

        temp_performance = pd.DataFrame([(str(round(score_train[1], 4)), str(round(score_val[1], 4)),
                                          str(round(score[1], 4)),
                                          window_size,
                                          final_epoch,
                                          parameters, current_time)], columns=performance_labels)
        performance_data = pd.concat([performance_data, temp_performance])
        performance_data.to_csv(save_file)

    # Need to re-standarize the original Y_test data to train the next model
    if standardize_output == 1:
        for i in range(Y_test.shape[1]):
            Y_test[:, i] = np.squeeze(scalers_output[i].transform(Y_test[:, i].reshape(-1, 1)))

    gc.collect()
    # clearing the backend keras session helps to avoid memory conflicts for the GPU when overwriting 'model'
    K.clear_session()

# Print the metric for the best model
logging.info(str(metric_comparison))

# Save best model as tflite model
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model_to_save)
tflite_model = converter.convert()

# Write the best model to a tflite file
with open(model_filename, 'wb') as f:
    f.write(tflite_model)
