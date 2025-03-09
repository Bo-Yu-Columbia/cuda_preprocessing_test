import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import unittest

# Define a minimal set of CUDA kernels for testing.

kernel_code = r"""
extern "C" {

__global__ void CropKernel(
    unsigned char *outBuffer,
    unsigned char *inBuffer,
    unsigned int cropWidth,
    unsigned int cropHeight,
    unsigned int inPitch,
    unsigned int channels,
    unsigned int offsetX,
    unsigned int offsetY)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cropWidth && row < cropHeight)
    {
        unsigned int inRow = row + offsetY;
        unsigned int inCol = col + offsetX;
        for (unsigned int c = 0; c < channels; c++)
        {
            outBuffer[row * cropWidth * channels + col * channels + c] =
                inBuffer[inRow * inPitch + inCol * channels + c];
        }
    }
}

__global__ void ResizeKernel(
    unsigned char *outBuffer,
    unsigned char *inBuffer,
    unsigned int inWidth,
    unsigned int inHeight,
    unsigned int inPitch,
    unsigned int outWidth,
    unsigned int outHeight,
    unsigned int channels)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < outWidth && row < outHeight)
    {
        float scaleX = (float)inWidth / outWidth;
        float scaleY = (float)inHeight / outHeight;
        unsigned int inCol = (unsigned int)(col * scaleX);
        unsigned int inRow = (unsigned int)(row * scaleY);
        for (unsigned int c = 0; c < channels; c++)
        {
            outBuffer[row * outWidth * channels + col * channels + c] =
                inBuffer[inRow * inPitch + inCol * channels + c];
        }
    }
}

__global__ void NormalizeKernel(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int inPitch,
    unsigned int channels,
    float scale,
    float *meanBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width && row < height)
    {
        for (unsigned int c = 0; c < channels; c++)
        {
            unsigned int idx = row * width * channels + col * channels + c;
            outBuffer[idx] = scale * ((float) inBuffer[row * inPitch + col * channels + c] - meanBuffer[idx]);
        }
    }
}

} // extern "C"
"""

mod = SourceModule(kernel_code)
crop_kernel = mod.get_function("CropKernel")
resize_kernel = mod.get_function("ResizeKernel")
normalize_kernel = mod.get_function("NormalizeKernel")

# Use a block dimension of 16x16 threads.
block_dim = (16, 16, 1)

class TestCudaKernels(unittest.TestCase):

    def test_crop_kernel(self):
        # Test: Crop a 4x3 region from a 6x6 image.
        in_height, in_width, channels = 6, 6, 3
        crop_offsetX, crop_offsetY = 1, 2
        crop_width, crop_height = 4, 3

        # Create a synthetic input image with sequential values.
        in_image = np.arange(in_height * in_width * channels, dtype=np.uint8)
        in_image = in_image.reshape((in_height, in_width, channels))
        # Expected result from Python slicing.
        expected = in_image[crop_offsetY:crop_offsetY+crop_height, crop_offsetX:crop_offsetX+crop_width, :]

        in_pitch = np.uint32(in_width * channels)
        in_gpu = cuda.mem_alloc(in_image.nbytes)
        cuda.memcpy_htod(in_gpu, in_image)
        out_image = np.empty((crop_height, crop_width, channels), dtype=np.uint8)
        out_gpu = cuda.mem_alloc(out_image.nbytes)

        grid_dim = ((crop_width + block_dim[0] - 1) // block_dim[0],
                    ((crop_height + block_dim[1] - 1) // block_dim[1]))

        crop_kernel(out_gpu, in_gpu,
                    np.uint32(crop_width), np.uint32(crop_height),
                    in_pitch, np.uint32(channels),
                    np.uint32(crop_offsetX), np.uint32(crop_offsetY),
                    block=block_dim, grid=grid_dim)

        cuda.memcpy_dtoh(out_image, out_gpu)
        self.assertTrue(np.array_equal(out_image, expected),
                        "Crop kernel output does not match expected result.")

    def test_resize_kernel(self):
        # Test: Resize a 4x4 image down to 2x2.
        in_height, in_width, channels = 4, 4, 3
        out_height, out_width = 2, 2

        in_image = np.arange(in_height * in_width * channels, dtype=np.uint8)
        in_image = in_image.reshape((in_height, in_width, channels))
        # Expected result using nearest-neighbor: take pixel at floor(row*scale, col*scale)
        expected = np.empty((out_height, out_width, channels), dtype=np.uint8)
        scaleX = in_width / out_width
        scaleY = in_height / out_height
        for row in range(out_height):
            for col in range(out_width):
                in_row = int(row * scaleY)
                in_col = int(col * scaleX)
                expected[row, col, :] = in_image[in_row, in_col, :]

        in_pitch = np.uint32(in_width * channels)
        in_gpu = cuda.mem_alloc(in_image.nbytes)
        cuda.memcpy_htod(in_gpu, in_image)
        out_image = np.empty((out_height, out_width, channels), dtype=np.uint8)
        out_gpu = cuda.mem_alloc(out_image.nbytes)

        grid_dim = ((out_width + block_dim[0] - 1) // block_dim[0],
                    ((out_height + block_dim[1] - 1) // block_dim[1]))

        resize_kernel(out_gpu, in_gpu,
                      np.uint32(in_width), np.uint32(in_height),
                      in_pitch,
                      np.uint32(out_width), np.uint32(out_height),
                      np.uint32(channels),
                      block=block_dim, grid=grid_dim)

        cuda.memcpy_dtoh(out_image, out_gpu)
        self.assertTrue(np.array_equal(out_image, expected),
                        "Resize kernel output does not match expected result.")

    def test_normalize_kernel(self):
        # Test: Normalize a simple 2x2 image with a zero mean buffer.
        height, width, channels = 2, 2, 3
        in_image = np.array([[[10, 20, 30],
                              [40, 50, 60]],
                             [[70, 80, 90],
                              [100, 110, 120]]], dtype=np.uint8)
        # With a scale factor of 1.0 and zero mean, the output should equal the input (as floats).
        scale = np.float32(1.0)
        mean_buffer = np.zeros((height, width, channels), dtype=np.float32).flatten()
        expected = in_image.astype(np.float32)

        in_pitch = np.uint32(width * channels)
        in_gpu = cuda.mem_alloc(in_image.nbytes)
        cuda.memcpy_htod(in_gpu, in_image)
        mean_gpu = cuda.mem_alloc(mean_buffer.nbytes)
        cuda.memcpy_htod(mean_gpu, mean_buffer)

        out_image = np.empty((height, width, channels), dtype=np.float32)
        out_gpu = cuda.mem_alloc(out_image.nbytes)

        grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
                    ((height + block_dim[1] - 1) // block_dim[1]))

        normalize_kernel(out_gpu, in_gpu,
                         np.uint32(width), np.uint32(height),
                         in_pitch,
                         np.uint32(channels),
                         scale,
                         mean_gpu,
                         block=block_dim, grid=grid_dim)

        cuda.memcpy_dtoh(out_image, out_gpu)
        self.assertTrue(np.allclose(out_image, expected),
                        "Normalize kernel output does not match expected result.")

if __name__ == '__main__':
    unittest.main()
