#! /usr/bin/env python

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

# Node specific imports
from datetime import datetime
import pynwb
import struct
import shutil
import time
import subprocess
import tempfile
from pathlib import Path
import socket


## set up logging (same for all derivatives)
NAME = 'save_nwb'
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


## Allows for testing what parameters are needed per derivative
testing = False
if testing:
    parameters_command_line = dict()
    parameters_command_line['full_rdb_save_path'] = '../Data/sim/2024-07-31/RawData/RDB/sim_240731T1035_EMG_read_process_train_step2.rdb'
    parameters_command_line['host_ip'] = '10.157.174.222'
    parameters_command_line['host_port'] = 50000

    # set all node-specific parameters that will be set within graph yaml file
    parameters = dict()

    # Stream specifics
    parameters['stream_definitions'] = '../brand-modules/emg-emulator-dl-module/stream_definitions.yaml'

    # Other node-specific parameters
    parameters['output_nwb_filename'] = 'output.nwb'
    
    # Convert to format recieved from Supervisor
    parameters = json.dumps(parameters)

else:
    parameters_command_line = dict()
    parameters_command_line['full_rdb_save_path'] = sys.argv[1]
    parameters_command_line['host_ip'] = sys.argv[2]
    parameters_command_line['host_port'] = sys.argv[3]
    parameters = sys.argv[4]

parameters = json.loads(parameters)
redis_host = parameters_command_line['host_ip']
redis_port = parameters_command_line['host_port']

# Load stream definitions
with open(parameters['stream_definitions'], 'r') as file:
    stream_definitions = yaml.safe_load(file)
output_streams = stream_definitions['RedisStreams']['Outputs']

# Get the path of the saved rdb file
source_path = parameters_command_line['full_rdb_save_path']

# Get the path for the nwb_saving_folder
# This folder is used as a temporary folder to store a temporary dump.rdb file
# This is necessary to load the old rdb as you won't accidentally destroy the rdb backup file
script_dir = os.path.dirname(os.path.abspath(__file__))
new_working_directory = Path(os.path.join(script_dir, 'nwb_saving_folder'))

# Log the location of the new working directory
logging.info(new_working_directory)

# Used to create temporary redis server
local_port = 6380

# Start a new redis server to avoid screwing up the backup rdb file
def start_redis_server(port=local_port, data_dir=new_working_directory):
    try:
        # Start Redis server process
        redis_server_process = subprocess.Popen(['redis-server', '--port', str(port), '--dir', data_dir, '--appendonly', 'no'])
        time.sleep(5)  # Wait for the server to start
        logging.info("Redis server started successfully.")
        return redis_server_process
    except Exception as e:
        logging.info(f"Error starting Redis server: {e}")
        return None

# It was hard to get a redis server to stop if it wasn't closed correctly
# In the following functions, we instead find the pid of the running server (if it exists) and then we terminate it
# We are only ever shutting down the local redis server, and not the supervisor's redis server
def stop_redis_server(process):
    if process:
        process.terminate()
        logging.info("Redis server stopped.")

def find_redis_pid(port):
    try:
        # Use netstat to find the process ID of the Redis server
        netstat_output = subprocess.check_output(['netstat', '-tulnp']).decode('utf-8')
        lines = netstat_output.split('\n')
        for line in lines:
            if f':{port}' in line and 'LISTEN' in line:
                parts = line.split()
                pid = None
                # Ensure we parse the line correctly to extract the PID
                for part in parts:
                    if '/' in part:
                        pid_str = part.split('/')[0]
                        if pid_str.isdigit():
                            pid = int(pid_str)
                            break
                if pid:
                    return pid
    except Exception as e:
        logging.error(f"Error finding Redis PID on port {port}: {e}")
    return None

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def force_kill_process(pid):
    try:
        os.kill(pid, signal.SIGKILL)
        logging.info(f"Forcefully killed process with PID {pid}.")
    except Exception as e:
        logging.error(f"Error forcefully killing process with PID {pid}: {e}")

def shutdown_redis_by_pid(port):
# TODO add docs
    pid = find_redis_pid(port)
    if pid:
        force_kill_process(pid)
        # Verify that the port is free
        if not is_port_in_use(port):
            logging.info(f"Port {port} is now free.")
        else:
            logging.warning(f"Port {port} is still in use after forceful termination.")
    else:
        logging.info(f"No Redis server found on port {port}, so no need to shutdown.")

# Copy rdb file to the new working directory
shutil.copy(source_path, new_working_directory / Path('dump.rdb'))
os.chmod(new_working_directory / Path('dump.rdb'), 0o777)

# In case a previous script was interupted and the redis server wasn't shut  down correctly
# Shutting down through redis caused problems,  so I switched to shutting down PID
#shutdown_redis_server(host='localhost', port=local_port)
shutdown_redis_by_pid(port=local_port)

# Start  the redis  server
redis_server_process = start_redis_server(port=local_port, data_dir=new_working_directory)

# Connect to the new local redis server
try:
    logging.info(f"Connecting to Redis at 'localhost':{local_port}...")
    r = Redis('127.0.0.1', local_port, retry_on_timeout=True)
    r.ping()
except ConnectionError as e:
    logging.error(f"Error with Redis connection, check again: {e}")
    sys.exit(1)
except:
    logging.error('Failed to connect to Redis. Exiting.')
    sys.exit(1)
logging.info('Redis connection successful.')

# Set or create new NWB directory
output_nwb_filename = parameters['output_nwb_filename']
if os.path.dirname(output_nwb_filename):
    dir_path = os.path.dirname(output_nwb_filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Initialize NWB file
nwbfile = pynwb.NWBFile('Neural Data', 'EXAMPLE_ID', datetime.now().astimezone())

# Create a device
device = nwbfile.create_device(name='Device')

# Create an electrode group (necessary for NWB files)
electrode_group = nwbfile.create_electrode_group(
    name='Electrode Group',
    description='description',
    location='location',
    device=device)

# This will loop through the provided stream list and all stream data will be saved together in one nwb file
for stream in list(output_streams.keys()):

    stream_info = output_streams[stream]

    if stream_info['enable_nwb'] == True:
        input_stream = stream

        # Read from stream
        stream_values = r.xrange(input_stream, min='-', max='+')
        n_channels = stream_info['samples']['chan_per_stream']
        frequency = stream_info['samples']['frequency']
        refresh_freq = stream_info['samples']['refresh_freq']
        window_size = frequency// refresh_freq

        input_characteristic = stream_info['samples']['nwb']['unit']

        # Different cases for different types of data (EMG, invasive, force, experimental, etc)
        if stream_info['type_nwb'] == 'TimeSeries':

            # Add electrodes to the electrode table
            for i in range(n_channels):
                nwbfile.add_electrode(
                    id=i,
                    x=np.nan, y=np.nan, z=np.nan,  # Add actual coordinates if available
                    imp=np.nan,  # Add impedance if available
                    location='location',
                    filtering='filtering',
                    group=electrode_group
                )

            # Create an electrode table region for the electrical series
            electrode_table_region = nwbfile.create_electrode_table_region(
                region=list(range(n_channels)),
                description='All electrodes'
            )

            # Initialize data and time matrices
            timestamps = np.zeros(((np.shape(stream_values)[0]-1)*window_size, ),  dtype=np.float64)
            data = np.zeros(((np.shape(stream_values)[0]-1)*window_size, n_channels),  dtype=stream_info['samples']['sample_type'])

            logging.info(f'Stream read')

            count = 0
            for entry in stream_values:

                _, fields = entry

                sample_bytes = fields[input_characteristic.encode()]

                if fields[b'ts_end'] == b'0.0':
                    t_init = float(fields[b'ts'].decode())
                    continue
                else:

                    ts_0 = t_init
                    ts_end = float(fields[b'ts_end'].decode())

                    sample_data = np.frombuffer(sample_bytes, dtype=stream_info['samples']['sample_type'])

                    # Flatten the data array to (num_samples * window_size, n_channels)
                    flattened_data = sample_data.reshape(-1, n_channels)

                    # Create timestamps for each timepoint
                    temp_timestamps = np.linspace(ts_0, ts_end, flattened_data.shape[0])

                    timestamps[count*window_size:(count+1)*window_size] = temp_timestamps

                    data[count*window_size:(count+1)*window_size, :] = flattened_data


                    t_init = float(fields[b'ts'].decode())
                    count = count + 1


            logging.info(f'Finished collecting data')

            # Create ElectricalSeries
            spiking_data = pynwb.ecephys.ElectricalSeries(
                    name=stream,
                    data=data,
                    electrodes=electrode_table_region,
                    timestamps=timestamps,
                    description=stream_info['samples']['nwb']['description']
                )

            nwbfile.add_acquisition(spiking_data)

            del spiking_data, flattened_data, sample_data, data, t_init, timestamps, temp_timestamps

        elif stream_info['type_nwb'] == 'Position':

            # Initialize data and time matrices
            timestamps = np.zeros(((np.shape(stream_values)[0] - 1) * window_size,), dtype=np.float64)
            data = np.zeros(((np.shape(stream_values)[0] - 1) * window_size, n_channels), dtype=stream_info['samples']['sample_type'])

            logging.info(f'Stream read')
            count=0

            for entry in stream_values:
                _, fields = entry

                sample_bytes = fields[input_characteristic.encode()]


                if fields[b'ts_end'] == b'0.0':
                    t_init = float(fields[b'ts'].decode())
                    continue
                else:

                    ts_0 = t_init
                    ts_end = float(fields[b'ts_end'].decode())

                    sample_data = np.frombuffer(sample_bytes, dtype=stream_info['samples']['sample_type'])


                    # Flatten the data array to (num_samples * window_size, n_channels)
                    flattened_data = sample_data.reshape(-1, n_channels)

                    # Create timestamps for each timepoint
                    temp_timestamps = np.linspace(ts_0, ts_end, flattened_data.shape[0])


                    timestamps[count * window_size:(count + 1) * window_size] = temp_timestamps
                    data[count * window_size:(count + 1) * window_size, :] = flattened_data

                    t_init = float(fields[b'ts'].decode())
                    count = count + 1


            position = pynwb.behavior.Position(name=stream)
            
            temp_position = pynwb.behavior.SpatialSeries(
                name=f'{stream}',
                data=data,
                reference_frame='near 0 is low  level force',
                timestamps=timestamps,
                description=stream_info['samples']['nwb']['description']
                )
            
            position.add_spatial_series(temp_position)

            # Add the Position object to the NWB file
            nwbfile.add_acquisition(position)

        else:
            logging.info(f'Unrecognized stream type')

# Write to NWB file
with pynwb.NWBHDF5IO(output_nwb_filename, 'w') as io:
    io.write(nwbfile)
os.chmod(output_nwb_filename, 0o777)
logging.info(f'NWB file saved to {output_nwb_filename}')

# Kill temporary redis server
stop_redis_server(redis_server_process)
