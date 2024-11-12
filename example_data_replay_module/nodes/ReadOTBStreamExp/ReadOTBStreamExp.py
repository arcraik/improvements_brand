#!/usr/bin/env python
# -*- coding: utf-8 -*-
# nodeTemplateBRAND.py

# Libraries for most Brand nodes
from brand import BRANDNode
import logging
import time
import gc
import numpy as np

# Necessary if you are having trouble getting supervisor to correctly shut down node
import os
import signal

# For testing prior to compilation
import redis
# import pdb

# Node specific imports
import pandas as pd
from typing import Tuple
from pathlib import Path
import tempfile
import tarfile
import xmltodict
import ast
from scipy import signal
import json


def import_otb(filename: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Imports .otb files and returns data matrix with channel timeseries and
    their corresponding metadata

    Parameters
    ----------
    filename : str
        `.otb` filename to be imported

    Returns
    -------
    data: np.ndarray, (n_channels, n_samps)
        Time-serieses (n_channels, n_samps)
    channel_df: pd.DataFrame
        Channel metadata
    """
    # Unzip file to temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    with tarfile.open(filename) as tar:
        print('Imput file: {}'.format(filename))
        tar.extractall(path=temp_dir.name)
        print('Extracted .otb file in temporary dir!')

    # Load session metadata
    with open(temp_path / 'patient.xml') as fd:
        # Avoid @ for attributes
        session = xmltodict.parse(fd.read(), attr_prefix='')

    # Read device and acquition specific metadata
    sigs = list(temp_path.rglob('*.sig'))
    if len(sigs) > 1:
        raise ('More than 1 .sig files were found, check ... This code expects one sig file per .otb file')

    abstract_fname = sigs[0].stem + '.xml'
    with open(temp_path / abstract_fname) as fd:
        # Avoid @ for attributes
        abst = xmltodict.parse(fd.read(), attr_prefix='')
        device = abst['Device']
        PowerSupply = 5

        # ensure channel metadata are in homogenious format i.e. list[dict]
        for i in range(len(device['Channels']['Adapter'])):
            # case of single channel
            if isinstance(device['Channels']['Adapter'][i]['Channel'], dict):
                device['Channels']['Adapter'][i]['Channel'] = [
                    device['Channels']['Adapter'][i]['Channel']]

        # convert device's arithmetic string attributes to numbers
        for key, value in device.items():
            if isinstance(value, str) and value.isnumeric():
                # string to int/float, safely
                device[key] = ast.literal_eval(value)

    # Extract channel metadata
    # 1. extending it on channel level, 2. apply simple DF operations

    # Data import and type casting
    channel_df = pd.DataFrame(device['Channels']['Adapter']).explode(
        'Channel').reset_index(drop=True)
    channel_df = channel_df.astype({'Gain': 'float',
                                    'HighPassFilter': 'float',
                                    'LowPassFilter': 'float',
                                    'AdapterIndex': 'int',
                                    'ChannelStartIndex': 'int'})

    # Extract adapter specific information to the channel
    adapter_info = channel_df.apply(lambda row:
                                    {'connector': row['Channel']['Prefix'],
                                     'sensor': row['Channel']['ID'],
                                     'sensor_description': row['Channel']['Description'],
                                     # relative index
                                     'rel_index': int(row['Channel']['Index']),
                                     'side': row['Channel']['Side'],
                                        'muscle': row['Channel']['Muscle']
                                     },
                                    axis='columns', result_type='expand')

    channel_df = pd.concat([channel_df, adapter_info], axis='columns')
    channel_df.drop('Channel', axis=1, inplace=True)

    # Detect HDsEMG array number
    tmp = channel_df['connector'].str.findall(r"MULTIPLE IN ([0-9]{1})")
    channel_df['array_num'] = tmp.apply(
        lambda x:  int(x[0]) if len(x) == 1 else None)

    # Make sure channel indeces are correct
    channel_df['channel_index'] = channel_df['ChannelStartIndex'] + \
        channel_df['rel_index']
    channel_df.set_index('channel_index', verify_integrity=True)

    # add sampling frequency and Digital to Analog multiplier
    channel_df['fsamp'] = device['SampleFrequency']
    # d2a_fn = lambda Vref, nAD, gain: Vref/(2**nAD)*1e6/gain
    channel_df['dig2analog_factor'] = channel_df['Gain'].apply(
        lambda G: PowerSupply/(2**device['ad_bits'])*1e3/G)

    # rename columns
    channel_df.rename(columns={'ID': 'adapter',
                               'Gain': 'gain',
                               'HighPassFilter': 'HP_filter',
                               'LowPassFilter': 'LP_filter'},
                      inplace=True)

    # extract only the relative
    channel_meta_df = channel_df[['channel_index', 'array_num', 'rel_index',
                                  'sensor', 'adapter', 'gain', 'HP_filter', 'LP_filter',
                                  'fsamp', 'dig2analog_factor',
                                  'side', 'muscle', 'connector']]

    """
    Load raw signal
    """
    # h=fopen(fullfile('tmpopen',signals(nSig).name),'r');
    # data=fread(h,[nChannel{nSig} Inf],'short');
    # binary file read: https://stackoverflow.com/a/14720675
    n_channels = len(channel_meta_df)
    with open(sigs[0], 'rb') as fd:
        data = np.fromfile(fd, np.int16).reshape((-1, n_channels)).T

    d2a_factor = channel_meta_df['dig2analog_factor'].to_numpy()

    # Digital to analog
    # multiply vector elements, row
    data = np.einsum('i,ij->ij', d2a_factor, data)

    temp_dir.cleanup()  # delete temporary folder

    return data, channel_meta_df

class ReadOTBStreamExp(BRANDNode):

    def __init__(self):

        # Inherent all attributes
        super().__init__()

        # Parameters for streaming
        if 'sync_key' in self.parameters:
            self.sync_key = self.parameters['sync_key']
        self.output_stream_1 = self.parameters['output_stream_1']
        self.output_characteristic_1 = self.parameters['output_characteristic_1']
        self.input_stream = self.parameters['input_stream']
        self.data_type = self.parameters['data_type']
        self.max_len = self.parameters['max_len']
        self.desired_fs = self.parameters['desired_fs']
        self.experimental_feature = self.parameters['experimental_feature']
        self.refresh_freq = self.parameters['refresh_freq']

        # Node-specific parameters
        self.otb_filepath = self.parameters['otb_filepath']
        self.testing = self.parameters['testing']

        # Initialize timing variables for samples here
        self.sample = {
            'ts': float(),  # time at which the output is written
            'ts_end': float(),  # time at which XADD is complete
            b'sync': json.dumps(dict()),
        }

        # Initialize or load any data structures here
        logging.info(f'Loading data')
        self.dataset, self.channel_df = import_otb(self.otb_filepath)
        self.fsamp = self.channel_df.loc[0, 'fsamp']

        # Analog to Digital, and representation as int16
        d2a_factor = self.channel_df['dig2analog_factor'].to_numpy()  # digital to analog factor, per channnel
        self.dataset = np.einsum('i,ij->ij', 1 / d2a_factor, self.dataset)  # from mV to integers (multiply vector elements, row)
        self.dataset = self.dataset.astype(self.data_type)  # type casting to int16

        # Select experiment channels
        feature_channel = self.channel_df[self.channel_df['sensor'] == self.experimental_feature].index[0]
        self.dataset = self.dataset[feature_channel, :]
        
        # If only looking at one channel, the dataset becomes 1-D
        if np.size(feature_channel)==1:
            self.dataset = np.expand_dims(self.dataset, 0)

        logging.info(f'Resampling stream')
        self.dataset = signal.resample(self.dataset, int(np.shape(self.dataset)[1]/self.fsamp*self.desired_fs), axis=1)

        # Calculating window size (since this is a function of sampling rate and refresh frequency)
        self.n_samps_buffer = self.desired_fs // self.refresh_freq
        self.sel_samps_0 = np.arange(self.n_samps_buffer)

        # I think I had an issue with dataset becoming 1D again for some reason.. one last check
        if self.dataset.ndim == 1:
            self.dataset = np.expand_dims(self.dataset, 0)

        self.dataset_size = np.shape(self.dataset)[1]

        logging.info(f'Data loaded and processed')


    def run(self):

        logging.info(f'Node is about to start streaming')

        sel_samps = lambda i: self.sel_samps_0 + i * self.n_samps_buffer

        iter = 0
        ts_init = time.time()
        while True:

            # Wait until a sample comes in from the data replayer before sending the corresponding experimental data
            # This guarentees that the data is synchronized as originally collected
            try:
                self.sample_received = self.r.xread({self.input_stream: '$'}, count=1, block=6000)
            except redis.exceptions.TimeoutError:
                self.sample_received = None

            # This helps this node know when the EMG replayer finishes
            # This means the number of times to replay data is set with only the EMG replayer
            if not self.sample_received:
                break

            # Get time when data processing begins
            ts = ts_init
            self.sample['ts'] = ts

            # Set samples into dictionary
            self.sample[bytes(self.output_characteristic_1, 'utf-8')] = self.dataset[:, sel_samps(iter)].astype(self.data_type).tobytes()

            # Sync functionality
            if self.sync_key:
                incomping_sync = json.loads(self.sample_received[0][1][0][1][b'sync'])
                incomping_sync[self.sync_key] = time.time()
                self.sample[b'sync'] = json.dumps(incomping_sync)

            # Add sample to output stream
            self.r.xadd(self.output_stream_1, self.sample, maxlen=self.max_len, approximate=True)

            # Not much processesing is happening, so, for replaying, we set the end ts where it should be
            self.sample['ts_end'] = self.sample['ts'] + 1 / self.desired_fs

            # The new ts_init must take into account the number of samples per buffer and  the  refresh frequency
            ts_init = self.sample['ts_end'] +1/self.refresh_freq/self.n_samps_buffer

            iter = iter + 1
            if sel_samps(iter)[-1] > self.dataset_size:
                iter = 0

    def terminate(self, sig, frame):
        
        logging.info('Terminating node')

        # Point to terminate function from inherited node class
        super().terminate(sig, frame)

if __name__ == '__main__':

    gc.disable()

    read_otb_stream_exp = ReadOTBStreamExp()

    read_otb_stream_exp.run()

    logging.info(f'Node finished')

    gc.collect()
