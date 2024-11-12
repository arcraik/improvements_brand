#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ReadNWBDecode.py

# Libraries for most Brand nodes
from brand import BRANDNode
import logging
import time
import gc
import numpy as np
import json

# Necessary if you are having trouble getting supervisor to correctly shut down node
import os
import signal

# For testing prior to compilation
import redis
# import pdb

# For GPU nodes
import tensorflow as tf

class ReadStreamDecodeEMG(BRANDNode):

    def __init__(self):

        # Inherent all attributes
        super().__init__()

        # Paramters for streaming
        if 'sync_key' in self.parameters:
            self.sync_key = self.parameters['sync_key']
        self.input_stream = self.parameters['input_stream']
        self.input_characteristic = self.parameters['input_characteristic']
        self.output_stream = self.parameters['output_stream']
        self.output_characteristic = self.parameters['output_characteristic']
        self.max_len = self.parameters['max_len']
        self.input_data_type = self.parameters['input_data_type']
        self.output_data_type = self.parameters['output_data_type']

        # Node-specific parameters
        self.model_file_path = self.parameters['model_file_path']
        self.refresh_freq = self.parameters['refresh_freq']
        self.desired_fs = self.parameters['desired_fs']
        self.window_size = self.desired_fs // self.refresh_freq


        # Initialize timing variables for sample here
        self.sample = {
            'ts_start': float(),  # time at which we start XREAD
            'ts_in': float(),  # time at which the input is received
            'ts': float(),  # time at which the output is written
            'ts_end': float(),  # time at which XADD is complete
            'i': int(), # not sure what these do, maybe when having to process several samples at once
            'i_in': int(),
            b'sync': json.dumps(dict()),
        }

        # Initialize deep learning model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.input_shape = self.input_details[0]['shape']
        self.output_details = self.interpreter.get_output_details()

    def run(self):

        logging.info(f'Node is about to start streaming')

        while True:

            # Get time when reading data begins
            self.sample['ts_start'] = time.time()

            try:
                self.sample_received = self.r.xread({self.input_stream: '$'}, count=1, block=6000)
            except redis.exceptions.TimeoutError:
                self.sample_received = None

            if not self.sample_received:
                break

            # This gives you a dictionary for hand_vel streams
            self.data_to_decode = np.frombuffer(self.sample_received[0][1][0][1][bytes(self.input_characteristic, 'utf-8')], dtype=self.input_data_type)
            self.data_to_decode = np.reshape(self.data_to_decode, (int(len(self.data_to_decode) / self.window_size), self.window_size))
            self.data_to_decode = np.expand_dims(np.expand_dims(self.data_to_decode, 0), -1).astype(self.input_data_type)

            # Get time when data read ends
            self.sample['ts_in'] = time.time()

            self.interpreter.set_tensor(self.input_details[0]['index'], self.data_to_decode)
            self.interpreter.invoke()
            self.prediction = np.append(self.interpreter.get_tensor(self.output_details[0]['index']), [0]).astype(self.output_data_type)

            self.sample[bytes(self.output_characteristic, 'utf-8')] = self.prediction.tobytes()

            # Get time when data processing complete
            self.sample['ts'] = time.time()

            if self.sync_key:
                incoming_sync = json.loads(self.sample_received[0][1][0][1][b'sync'])
                incoming_sync[self.sync_key] = time.time()
                self.sample[b'sync'] = json.dumps(incoming_sync)

            # Add sample to output stream
            self.r.xadd(self.output_stream, self.sample, maxlen=self.max_len, approximate=True)

            # Get time when xadd complete
            self.sample['ts_end'] = time.time()


    def terminate(self, sig, frame):
        
        logging.info('Terminating Node')
        
        # Point to terminate function from inherited node class
        super().terminate(sig, frame)


if __name__ == '__main__':

    gc.disable()

    read_Stream_decodeEMG = ReadStreamDecodeEMG()

    read_Stream_decodeEMG.run()

    gc.collect()