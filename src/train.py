from utils.ffn_utils import make_ffn, export_ffn_to_onnx

# how too build a FFN
ffn = make_ffn([
    (10, 20, "relu"),
    (20, 5, "softmax")
])

# How too export this to a ONNX file
export_ffn_to_onnx(ffn, input_size=10, filename="fnn.onnx")
