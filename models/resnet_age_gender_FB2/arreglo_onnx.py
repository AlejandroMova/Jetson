import onnx
from onnx import helper, TensorProto

model = onnx.load("models/resnet_age_gender_FB2/resnet18_finetuned_int8_fixed.onnx")

# Inspeccionar el nombre del output tensor antes de modificar
print("Outputs actuales:")
for o in model.graph.output:
    print(" ", o.name)

# Asume que el output se llama "output" (igual que FB1 — verificar con el print arriba)
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

onnx.save(model, "models/resnet_age_gender_FB2/resnet18_finetuned_int8_fixed.onnx")
print("Modelo sobreescrito con softmax. Borra el .engine para que rebuild.")
