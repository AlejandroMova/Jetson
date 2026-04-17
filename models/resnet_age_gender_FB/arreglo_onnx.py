import onnx
from onnx import helper, TensorProto

model = onnx.load("models/resnet_age_gender_FB/resnet18_age_gender_FB_int8.onnx")

# El output actual se llama "output" (confirmado con strings anteriormente)
softmax_node = helper.make_node(
    "Softmax",
    inputs=["output"],
    outputs=["output_prob"],
    axis=1   # dim de clases en tensor [batch, 6]
)
model.graph.node.append(softmax_node)

del model.graph.output[:]
model.graph.output.append(
    helper.make_tensor_value_info("output_prob", TensorProto.FLOAT, [None, 6])
)

onnx.save(model, "models/resnet_age_gender_FB/resnet18_age_gender_FB_int8.onnx")
print("Modelo sobreescrito con softmax. Borra el .engine para que rebuild.")