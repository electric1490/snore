
import os, pyaudio, time, logging

p = pyaudio.PyAudio()
os.system('clear')

print('Imported PyAudio...')

logging.basicConfig(
    filename="snore_app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M")

logging.warning("App Script Startup Initiated")

print('Loading Discord Webhook...')
from discord_webhook import DiscordWebhook

print('Loading TensorFlow...')
import numpy as np
import soundfile as sf
import tensorflow as tf

print('Loading YAMNet...')
import params as yamnet_params
import yamnet as yamnet_model

print('Loading YAMNet Model...')
params = yamnet_params.Params()
yamnet = yamnet_model.yamnet_frames_model(params)

print('Loading YAMNet Model Weights...')
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

print('Initializing PyAudio...')
CHUNK = 1024 # frames_per_buffer # samples per chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 7.024                                          # need at least 975 ms
INFERENCE_WINDOW = 2 * int(RATE / CHUNK * RECORD_SECONDS)       # 2 * 16 CHUNKs
THRESHOLD = 0.25

stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

print('YAMNET now detecting...')

webhook = DiscordWebhook(url="https://discord.com/api/webhooks/1337995743784599592/5kUrfH6NiMHQMVUrNgG3C3VqXa3KPpLbrtJE2PD8P1NkvB5Q4buoVGFvZ7wdB-g7wR00", rate_lmit_retry=True, content="RPi Snore Detection = Online")
response = webhook.execute()
logging.warning("Snore App Listening Loop Successfully Started")

CHUNKs = []

with open('sed.npy', 'ab') as f:
    while True:
        try:
            stream.start_stream()
            
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                CHUNKs.append(data)
                
            stream.stop_stream()

            if len(CHUNKs) > INFERENCE_WINDOW:
                CHUNKs = CHUNKs[int(RATE / CHUNK * RECORD_SECONDS):]
                
            wav_data = np.frombuffer(b''.join(CHUNKs), dtype=np.int16)
            waveform = wav_data / tf.int16.max#32768.0
            waveform = waveform.astype('float32')
            scores, embeddings, spectrogram = yamnet(waveform)
            prediction = np.mean(scores[:-1], axis=0) # last one scores comes from insufficient samples

            assert len(scores[:-1]) == CHUNK * len(CHUNKs) / RATE // 0.48 - 1 # hop 0.48 seconds
            top5 = np.argsort(prediction)[::-1][:5]
            if 38 in top5:
                webhook = DiscordWebhook(url="https://discord.com/api/webhooks/1337995743784599592/5kUrfH6NiMHQMVUrNgG3C3VqXa3KPpLbrtJE2PD8P1NkvB5Q4buoVGFvZ7wdB-g7wR00", rate_lmit_retry=True, content="Snore Alert - Keep it Down!")
                response = webhook.execute()
                print(time.ctime().split()[3],''.join((f" {prediction[i]:.2f} {yamnet_classes[i][:7].ljust(7, '　')}" if prediction[i] >= THRESHOLD else '') for i in top5))
                logging.warning("Snore Detected - Discord Webhook Sent")
                
            else:
                print(time.ctime().split()[3],''.join((f" {prediction[i]:.2f} {yamnet_classes[i][:7].ljust(7, '　')}" if prediction[i] >= THRESHOLD else '') for i in top5))

            np.save(f, np.concatenate(([time.time()], prediction)))

        except:
            stream.stop_stream()
            stream.close()
            p.terminate()
            f.close()

