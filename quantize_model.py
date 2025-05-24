from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/model.onnx",
    model_output="models/fashion_clip_int8.onnx",
    weight_type=QuantType.QUInt8 #chnage QInt8 to QUInt8
)
