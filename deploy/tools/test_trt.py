import tensorrt as trt
import numpy as np
import ctypes

cudart = ctypes.CDLL('libcudart.so')
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
# Disable cuDNN (Cask) tactics — they fail on Jetson TRT 10.3 with this model.
# Force cuBLAS + cuBLAS-LT only, which are reliable on Jetson unified memory.
config.set_tactic_sources(
    1 << int(trt.TacticSource.CUBLAS) |
    1 << int(trt.TacticSource.CUBLAS_LT)
)
print('FP32 mode, cuBLAS tactics only')
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open('/nx_tech/models/osnet/osnet_x1_0_market1501.onnx', 'rb') as f:
    parser.parse(f.read())
profile = builder.create_optimization_profile()
profile.set_shape('input', (1, 3, 256, 128), (1, 3, 256, 128), (1, 3, 256, 128))
config.add_optimization_profile(profile)
print('Construyendo engine (~30s)...')
serialized = builder.build_serialized_network(network, config)
print(f'Engine: {len(bytes(serialized)) / 1024:.0f} KB')
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(bytes(serialized))
context = engine.create_execution_context()
stream = ctypes.c_void_p()
cudart.cudaStreamCreate(ctypes.byref(stream))
inp = np.random.randn(1, 3, 256, 128).astype(np.float32)
out = np.zeros((1, 512), dtype=np.float32)
context.set_tensor_address('input', inp.ctypes.data)
context.set_tensor_address('output', out.ctypes.data)
context.execute_async_v3(stream.value)
cudart.cudaStreamSynchronize(stream.value)
print('Output shape:', out.shape, '| norm:', round(float(np.linalg.norm(out)), 3))
print('GPU inference OK')
