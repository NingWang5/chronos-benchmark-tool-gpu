import tensorrt as trt
import os
import pycuda.driver as cuda
from utils_data import generate_data


class Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, data, cache_file=None):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = '/root/wangning/benchmark/cache_file'

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = data

        input_sample = None
        for x, _ in self.data:
            input_sample = x.numpy()
            break

        self.batch_size = input_sample.shape[0]

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(input_sample.nbytes)

        self.num = 0
        self.batcher = self.create_batcher()

    def get_batch_size(self):
        return self.batch_size
    
    def create_batcher(self):
        for x, _ in self.data:
            yield x.numpy().ravel()

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            batch = next(self.batcher)
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except:
            raise StopIteration

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)