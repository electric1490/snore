import tensorflow as tf  # If you're using the TensorFlow Hub model directly
import numpy as np
import librosa
import pandas as pd

# 1. Load the YAMNet model (TensorFlow Hub or local .tflite)
# --- TensorFlow Hub ---
# model = tf.saved_model.load("https://tfhub.dev/google/yamnet/1") # For TensorFlow Hub
# --- Local TFLite Model ---
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path="yamnet.tflite")  # Path to your .tflite file
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Load the audio file
filepath = "audio_file.wav"  # Path to your audio file
y, sr = librosa.load(filepath, sr=16000)  # Load at 16kHz sample rate (YAMNet's expected rate)

# 3. Preprocess the audio (if needed)
# YAMNet expects audio in chunks of 1 second (16000 samples).  If your audio is longer,
# you'll need to split it. If it's shorter, you might need to pad it, or just use it as is.
# Here's an example of splitting:
chunk_length = 16000
num_chunks = int(np.ceil(len(y) / chunk_length))
predictions = []

for i in range(num_chunks):
    start = i * chunk_length
    end = min((i + 1) * chunk_length, len(y))
    audio_chunk = y[start:end]

    # Pad if the last chunk is shorter
    if len(audio_chunk) < chunk_length:
        audio_chunk = np.pad(audio_chunk, (0, chunk_length - len(audio_chunk)), 'constant')

    # Reshape for YAMNet (if it is a tflite model)
    audio_chunk = audio_chunk.reshape(1, chunk_length)

    # 4. Make predictions
    if "interpreter" in locals(): #If it is a tflite model
        interpreter.set_tensor(input_details[0]['index'], audio_chunk.astype(np.float32))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
    else: #If it is a TensorFlow Hub model
        prediction = model(audio_chunk)

    predictions.append(prediction)

predictions = np.array(predictions) #Convert predictions to a NumPy array

# 5. Load the class map
class_map_path = "yamnet_class_map.csv"  # Path to your class map CSV
class_map = pd.read_csv(class_map_path)

# 6. Process and interpret the predictions
for i, prediction in enumerate(predictions):
    # Get the top predicted class for each chunk
    top_class_index = np.argmax(prediction)
    predicted_class = class_map.iloc[top_class_index]['display_name']

    print(f"Chunk {i+1}: Predicted Class: {predicted_class}")

    # Or, if you want all the probabilities:
    # print(f"Chunk {i+1}: Predictions: {prediction}")

# If you want a single prediction for the entire audio file (if it is short enough or you want to combine the results):
if len(predictions) == 1:
    overall_top_class_index = np.argmax(predictions[0])
    overall_predicted_class = class_map.iloc[overall_top_class_index]['display_name']
    print(f"Overall Predicted Class: {overall_predicted_class}")
elif len(predictions) > 1: #If you want a prediction for the whole audio file and it is longer than 1 second
    #You can combine the results as you see fit. Here is a basic example
    average_predictions = np.mean(predictions, axis=0) #Averages the probabilities of each class for each chunk
    overall_top_class_index = np.argmax(average_predictions)
    overall_predicted_class = class_map.iloc[overall_top_class_index]['display_name']
    print(f"Overall Predicted Class (Averaged): {overall_predicted_class}")
