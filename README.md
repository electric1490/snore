How to Install the Snore App:

# Ensure that the Raspberry Pi is up to date.
sudo apt-get update
sudo apt-get upgrade
sudo reboot

# Upgrade pip first. Also make sure wheel is installed.
python -m pip install --upgrade pip wheel

# Create Virtual Environment
python -m venv snoreappENV
source snoreappENV/bin/activate

# Install dependences.
pip3 install numpy (appears to be version 2.0.2)
pip3 install resampy (appears to be version 0.4.3)
pip3 install tensorflow (appears to be version 2.18.0)
pip3 install soundfile (appears to be version 0.13.1)

# Clone TensorFlow models repo into a 'models' directory.
git clone https://github.com/tensorflow/models.git
cd models/research/audioset/yamnet
# Download data file into same directory as code.
curl -O https://storage.googleapis.com/audioset/yamnet.h5

# Installation ready, let's test it.
python yamnet_test.py
If we see "Ran 4 tests ... OK ...", then we're all set.
if it doesn't run because tf-keras was 3.8.0, it should downgrade to tf-keras
pip3 install tf-keras (appears to be version 2.18.0)
run python yamnet_test.py to determine if working
# Now let's install the other elements needed to run the program


pip3 install discord-webhook (appears to be version 1.3.1)
pip3 install pyaudio (appears to be version 0.2.14)
pip3 install matplotlib (appears to be version 3.10.0)
pip3 install librosa (appears to be version 0.10.2)

Load the 'yamnet_realtime_engine.py' into the same folder where the 'inference.py' file is placed. 
