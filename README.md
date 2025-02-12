# How to Install the Snore App:

### Ensure that the Raspberry Pi is up to date.
Working on a Raspberry Pi 4 running 64-bit OS (non-desktop)
```
sudo apt-get update
sudo apt-get upgrade (need to elect 'Y')
sudo reboot now
```

### Upgrade pip first. Also make sure wheel is installed.
```
sudo apt-get install python3-pip -y
sudo apt-get install git -y
sudo apt-get install portaudio19-dev -y
```

### Create Virtual Environment
```
python -m venv snoreappENV
source snoreappENV/bin/activate
```

### Install dependences.
Simplified PIP install: ``pip3 install numpy resampy tensorflow soundfile tf-keras``\
OR
```
pip3 install numpy            #*(appears to be version 2.0.2) (installed at 2.2.2)*
pip3 install resampy          #*(appears to be version 0.4.3) (uninstalled numpy and reinstalled numpy at 2.1.3)*
pip3 install tensorflow       #*(appears to be version 2.18.0) (installs numpy at 2.0.2)*
pip3 install soundfile        #*(appears to be version 0.13.1)*
pip3 install tf-keras         #*(appears to be version 2.18.0)*
```

### Clone TensorFlow models repo into a 'models' directory.
```
deactivate                                              #turns off the virtual environment
git clone https://github.com/electric1490/snore.git
git clone https://github.com/tensorflow/models.git      #this is a large file directly downloading the models from google
cd models/research/audioset/yamnet
```

### Download data file into same directory as code.
```
curl -O https://storage.googleapis.com/audioset/yamnet.h5
```

### Core Tensorflow application now ready for testing.
```
cd                                     #leave the yamnet folder
source snoreappENV/bin/activate        #reactivate the virtual environment
cd models/research/audioset/yamnet     #navigate to the yamnet folder
python yamnet_test.py                  #run this to determine if working
```
If we see "Ran 4 tests ... OK ...", then we're all set.\
if it doesn't run because tf-keras was 3.8.0, it should downgrade to tf-keras using ``pip3 install tf-keras``

### Now let's install the other elements needed to run the snore app
```
cd
pip3 install discord-webhook        #appears to be version 1.3.1
pip3 install pyaudio                #appears to be version 0.2.14
pip3 install matplotlib             #appears to be version 3.10.0
pip3 install librosa                #appears to be version 0.10.2
deactivate
```

### Lets add in the custom python files for the app
```
cp snore/yamnet_realtime_engine.py models/research/audioset/yamnet #load the 'yamnet_realtime_engine.py' into the same folder where the 'inference.py' file is placed.
cd models/research/audioset/yamnet                                 #navigate to the yamnet folder
python yamnet_realtime_engine.py                                   #run the model
```

### Lets make sure that the Raspberry Pi is using the correct hardware
Execute ``arecord -1`` to find the microphone card and device number\
Take note of the Card #, the device #, and the subdevice #\
navigate to ``cd /``\
then navigate to ``usr/share/alsa``\
then ``sudo nano alsa.conf`` and adjust the entry to match the output on ``arecord -l``\
example seen here:

```
#### output of running arecord -l in the terminal ####:

(snoreappENV) snore@snorelite:/usr/share/alsa $ arecord -l
**** List of CAPTURE Hardware Devices ****
card 1: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

####then set alsa.conf to the below:

defaults.ctl.card 1
defaults.pcm.card 1
defaults.pcm.device 0
defaults.pcm.subdevice 0
```
