#!/bin/bash

#place this in the same folder as the executable
#bash /home/snore/models/research/audioset/yamnet/startup_script.sh

source snoreappENV/bin/activate

python models/research/audioset/yamnet/yamnet_realtime_engine.py

