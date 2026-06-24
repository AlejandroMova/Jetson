import tensorrt as trt
import numpy as np
import ctypes

cudart = ctypes.CDLL('libcudart.so')
cudart.cudaSetDevice(0)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
# Level 0: TRT uses first-valid kernel per layer, avoids Cask selection
config.builder_optimization_level = 0
print('FP32, optimization_level=0', flush=True)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open('/nx_tech/models/osnet/osnet_x1_0_market1501.onnx', 'rb') as f:
    parser.parse(f.read())
profile = builder.create_optimization_profile()
profile.set_shape('input', (1, 3, 256, 128), (1, 3, 256, 128), (1, 3, 256, 128))
config.add_optimization_profile(profile)
print('Construyendo engine...', flush=True)
serialized = builder.build_serialized_network(network, config)
print(f'Engine: {len(bytes(serialized)) / 1024:.0f} KB', flush=True)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(bytes(serialized))
context = engine.create_execution_context()

stream = ctypes.c_void_p()
cudart.cudaStreamCreate(ctypes.byref(stream))

# Proper CUDA device memory — required for Cask/tensor-core kernels
IN_SIZE  = 1 * 3 * 256 * 128 * 4
OUT_SIZE = 1 * 512 * 4
inp_d = ctypes.c_void_p()
out_d = ctypes.c_void_p()
cudart.cudaMalloc(ctypes.byref(inp_d), ctypes.c_size_t(IN_SIZE))
cudart.cudaMalloc(ctypes.byref(out_d), ctypes.c_size_t(OUT_SIZE))

inp_h = np.random.randn(1, 3, 256, 128).astype(np.float32)
cudart.cudaMemcpy(inp_d, inp_h.ctypes.data,
                  ctypes.c_size_t(IN_SIZE), ctypes.c_uint(1))  # H→D

context.set_tensor_address('input',  inp_d.value)
context.set_tensor_address('output', out_d.value)
ok = context.execute_async_v3(stream.value)
cudart.cudaStreamSynchronize(stream.value)

out_h = np.zeros((1, 512), dtype=np.float32)
cudart.cudaMemcpy(out_h.ctypes.data, out_d,
                  ctypes.c_size_t(OUT_SIZE), ctypes.c_uint(2))  # D→H

cudart.cudaFree(inp_d)
cudart.cudaFree(out_d)

print('execute ok:', ok, flush=True)
print('Output shape:', out_h.shape, '| norm:', round(float(np.linalg.norm(out_h)), 3), flush=True)
print('GPU inference OK' if ok else 'FAILED', flush=True)
