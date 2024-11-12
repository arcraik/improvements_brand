These are modifications to [Backend for Real-time Asynchronous Neural Decoding (BRAND)](https://github.com/brandbci/brand). Specifically, these modifications enable automatic derivative operation, deep learning optimization, and data saving functionality. 
This repository also provides an example module that:

* Graph 1: Loads and replays EMG data (in OTB format), saves all data in NWB format, optimizes a CNN for EMG decoding, and saves the model for real-time inference.
* Graph 2: Loads and replays EMG data (in OTB format) and infers in real-time using the previously trained model

The repository includes stream definitions, graph files, the modified supervisor.py file, and the modified conda environment file.
