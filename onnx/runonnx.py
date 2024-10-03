import onnxruntime as ort
import numpy as np

# Ensure that ONNX Runtime can use GPU
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("CUDA is available")
    providers = ['CUDAExecutionProvider']
else:
    print("CUDA is not available, using CPU")
    providers = ['CPUExecutionProvider']

# Load the ONNX model with ONNX Runtime
session = ort.InferenceSession("./nconCSPN.onnx", providers=providers)

input_names = [input.name for input in session.get_inputs()]
print("Expected input names:", input_names)

# Prepare dummy inputs according to the expected input shapes
# These inputs should match the ones used during the export
dummy_rgb_input = np.random.randn(1, 3, 480, 640).astype(np.float32)
dummy_depth_input = np.random.randn(1, 1, 480, 640).astype(np.float32)
dummy_k = np.random.randn(1, 3, 3).astype(np.float32)

# Prepare the input dictionary; keys must match the input names used during the export
input_dict = {
    'rgb': dummy_rgb_input,
    'depth': dummy_depth_input,
    # 'k': dummy_k
}

# Run the model
outputs = session.run(None, input_dict)  # 'None' indicates that all outputs will be returned

# Extract outputs
output_depth = outputs[0]  # Assuming the first output is the depth map
output_confidence = outputs[1]  # Assuming the second output is the confidence map

# Print or process the outputs
print("Depth Map Output:", output_depth)
print("Confidence Map Output:", output_confidence)