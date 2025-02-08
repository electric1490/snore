import tflite_runtime.interpreter as tflite
import numpy as np
import sounddevice as sd  # For live audio input (install with: pip install sounddevice)
import librosa  # For audio processing (install with: pip install librosa)

# 1. Load the TFLite model
interpreter = tflite.Interpreter(model_path="yamnet.tflite") # Path to your YAMNet TFLite model
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# YAMNet expects audio input with a specific sample rate and shape
SAMPLE_RATE = 16000  # YAMNet's expected sample rate
INPUT_SHAPE = (1, 15600) # YAMNet's expected input shape (1 second of audio)

# 2. Audio Input (using sounddevice)
def get_audio_chunk(duration=1.0):  # Duration in seconds
    recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    return recording.reshape(INPUT_SHAPE)  # Reshape to match YAMNet input


# 3. Inference Loop
while True:
    audio_chunk = get_audio_chunk()

    # Preprocessing (if needed - YAMNet expects specific format)
    # audio_chunk = librosa.resample(audio_chunk.flatten(), SAMPLE_RATE, SAMPLE_RATE) # Resample if needed
    # audio_chunk = audio_chunk.reshape(INPUT_SHAPE) # Reshape after resampling

    interpreter.set_tensor(input_details[0]['index'], audio_chunk)
    interpreter.invoke()

    # 4. Get Predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Process the predictions (e.g., print the top predicted class)
    top_class = np.argmax(predictions)
    print(f"Predicted Class: {top_class}") # YAMNet's class indices

    # Or, get all the class probabilities:
    # print(predictions)

    # You can also use the YAMNet class map to get the actual class names
    # import pandas as pd
    # class_map = pd.read_csv("yamnet_class_map.csv") # Path to the class map CSV
    # predicted_class_name = class_map.iloc[top_class]['display_name']
    # print(f"Predicted Class Name: {predicted_class_name}")

# 5. Cleanup (if needed)
sd.terminate()
