import tflite_runtime.interpreter as tflite
import numpy as np

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a sample input
input_data = np.array(np.random.rand(*input_details[0]['shape']), dtype=np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)
