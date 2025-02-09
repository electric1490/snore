How to Install the Snore App:

# Ensure that the Raspberry Pi is up to date.
sudo apt-get update
sudo apt-get upgrade
sudo reboot

# Upgrade pip first. Also make sure wheel is installed.
python -m pip install --upgrade pip wheel

Install dependences.
pip3 install numpy resampy tensorflow soundfile

# Clone TensorFlow models repo into a 'models' directory.
git clone https://github.com/tensorflow/models.git
cd models/research/audioset/yamnet
# Download data file into same directory as code.
curl -O https://storage.googleapis.com/audioset/yamnet.h5

# Installation ready, let's test it.
python yamnet_test.py
# If we see "Ran 4 tests ... OK ...", then we're all set.

# Now let's install the other 
pip3 install tf-keras
pip3 install discord-webhook

Load the 'yamnet_realtime_engine.py' into the same folder where the 'inference.py' file is placed. 
