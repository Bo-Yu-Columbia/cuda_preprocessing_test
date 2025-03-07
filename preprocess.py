import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import unittest

import PyNvCodec as nvc

# CUDA kernels for cropping, resizing, padding, and normalization.
kernel_code = r"""
extern "C" {

__global__ void NvDsPreProcessCropKernel(
    unsigned char *outBuffer,
    unsigned char *inBuffer,
    unsigned int cropWidth,
    unsigned int cropHeight,
    unsigned int inPitch,
    unsigned int inputPixelSize,
    unsigned int offsetX,
    unsigned int offsetY)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cropWidth && row < cropHeight)
    {
        // Map output (crop) coordinates to input image coordinates.
        unsigned int inRow = row + offsetY;
        unsigned int inCol = col + offsetX;
        for (unsigned int k = 0; k < inputPixelSize; k++)
        {
            outBuffer[row * cropWidth * inputPixelSize + col * inputPixelSize + k] =
                inBuffer[inRow * inPitch + inCol * inputPixelSize + k];
        }
    }
}

__global__ void NvDsPreProcessResizeKernel(
    unsigned char *outBuffer,
    unsigned char *inBuffer,
    unsigned int inWidth,
    unsigned int inHeight,
    unsigned int inPitch,
    unsigned int outWidth,
    unsigned int outHeight,
    unsigned int inputPixelSize)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < outWidth && row < outHeight)
    {
        // Compute scaling ratios.
        float scaleX = (float)inWidth / outWidth;
        float scaleY = (float)inHeight / outHeight;
        // Nearest-neighbor mapping.
        unsigned int inCol = (unsigned int)(col * scaleX);
        unsigned int inRow = (unsigned int)(row * scaleY);
        for (unsigned int k = 0; k < inputPixelSize; k++)
        {
            outBuffer[row * outWidth * inputPixelSize + col * inputPixelSize + k] =
                inBuffer[inRow * inPitch + inCol * inputPixelSize + k];
        }
    }
}

__global__ void NvDsPreProcessPaddingKernel(
    unsigned char *outBuffer,
    unsigned char *inBuffer,
    unsigned int outWidth,
    unsigned int outHeight,
    unsigned int inWidth,
    unsigned int inHeight,
    unsigned int outPitch,
    unsigned int inPitch,
    unsigned int inputPixelSize,
    unsigned int offsetX,
    unsigned int offsetY,
    unsigned char padValue)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < outWidth && row < outHeight)
    {
        if (col >= offsetX && col < (offsetX + inWidth) &&
            row >= offsetY && row < (offsetY + inHeight))
        {
            // Map output coordinates to input coordinates.
            unsigned int inRow = row - offsetY;
            unsigned int inCol = col - offsetX;
            for (unsigned int k = 0; k < inputPixelSize; k++)
            {
                outBuffer[row * outPitch + col * inputPixelSize + k] =
                    inBuffer[inRow * inPitch + inCol * inputPixelSize + k];
            }
        }
        else
        {
            // Fill with pad value.
            for (unsigned int k = 0; k < inputPixelSize; k++)
            {
                outBuffer[row * outPitch + col * inputPixelSize + k] = padValue;
            }
        }
    }
}

// Normalization kernel: converts input (unsigned char) to float,
// reverses channel order (BGR->RGB), subtracts mean, and scales.
__global__ void NvDsPreProcessConvert_CxToL3RFloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width && row < height)
    {
        for (unsigned int k = 0; k < 3; k++)
        {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * ((float) inBuffer[row * pitch + col * inputPixelSize + (2 - k)] -
                meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

} // extern "C"
"""

# Compile CUDA kernels.
mod = SourceModule(kernel_code)
crop_kernel = mod.get_function("NvDsPreProcessCropKernel")
resize_kernel = mod.get_function("NvDsPreProcessResizeKernel")
padding_kernel = mod.get_function("NvDsPreProcessPaddingKernel")
normalize_kernel = mod.get_function("NvDsPreProcessConvert_CxToL3RFloatKernelWithMeanSubtraction")

# Set block dimensions.
block_dim = (16, 16, 1)

class TestPreprocessingKernels(unittest.TestCase):
    def test_crop_kernel(self):
        # Unit test for cropping using synthetic data.
        in_width, in_height, channels = 10, 10, 3
        in_image = np.arange(in_height * in_width * channels, dtype=np.uint8).reshape((in_height, in_width, channels))
        offsetX, offsetY = 2, 3
        crop_width, crop_height = 4, 4
        expected = in_image[offsetY:offsetY+crop_height, offsetX:offsetX+crop_width, :].copy()
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
                    np.uint32(offsetX), np.uint32(offsetY),
                    block=block_dim, grid=grid_dim)
        cuda.memcpy_dtoh(out_image, out_gpu)
        self.assertTrue(np.array_equal(out_image, expected), "Crop kernel output mismatch.")

    def test_resize_kernel(self):
        # Unit test for resizing using synthetic data.
        in_width, in_height, channels = 8, 8, 3
        in_image = np.random.randint(0, 256, (in_height, in_width, channels), dtype=np.uint8)
        out_width, out_height = 4, 4
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
        self.assertTrue(np.array_equal(out_image, expected), "Resize kernel output mismatch.")

    def test_padding_kernel(self):
        # Unit test for padding using synthetic data.
        in_width, in_height, channels = 5, 5, 3
        in_image = np.random.randint(0, 256, (in_height, in_width, channels), dtype=np.uint8)
        out_width, out_height = 10, 10
        offsetX, offsetY = 3, 2
        pad_value = np.uint8(0)
        expected = np.full((out_height, out_width, channels), pad_value, dtype=np.uint8)
        expected[offsetY:offsetY+in_height, offsetX:offsetX+in_width, :] = in_image
        in_pitch = np.uint32(in_width * channels)
        out_pitch = np.uint32(out_width * channels)
        in_gpu = cuda.mem_alloc(in_image.nbytes)
        cuda.memcpy_htod(in_gpu, in_image)
        out_image = np.empty((out_height, out_width, channels), dtype=np.uint8)
        out_gpu = cuda.mem_alloc(out_image.nbytes)
        grid_dim = ((out_width + block_dim[0] - 1) // block_dim[0],
                    ((out_height + block_dim[1] - 1) // block_dim[1]))
        padding_kernel(out_gpu, in_gpu,
                       np.uint32(out_width), np.uint32(out_height),
                       np.uint32(in_width), np.uint32(in_height),
                       out_pitch, in_pitch,
                       np.uint32(channels),
                       np.uint32(offsetX), np.uint32(offsetY),
                       pad_value,
                       block=block_dim, grid=grid_dim)
        cuda.memcpy_dtoh(out_image, out_gpu)
        self.assertTrue(np.array_equal(out_image, expected), "Padding kernel output mismatch.")

    def test_video_processing(self):
        video_path = "test_video.mp4"
        if not os.path.exists(video_path):
            self.skipTest("Video file {} not found.".format(video_path))
        gpu_id = 0
        nvDec = nvc.PyNvDecoder(video_path, gpu_id, "rgb")
        conv = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvDec.Format(), nvc.PixelFormat.RGB, gpu_id)

        # Preprocessing parameters:
        # Crop bounding box.
        crop_offsetX, crop_offsetY = 50, 50
        crop_width, crop_height = 200, 200
        target_width, target_height, channels = 128, 384, 3

        # Normalization parameters.
        scaleFactor = np.float32(1.0 / 255.0)
        # Create a mean data buffer; here we use a constant mean (128.0) per channel.
        mean_values = np.full((target_height, target_width, channels), 128.0, dtype=np.float32)
        mean_host = mean_values.flatten()
        mean_gpu = cuda.mem_alloc(mean_host.nbytes)
        cuda.memcpy_htod(mean_gpu, mean_host)

        processed_frames = 0
        max_frames = 10

        while processed_frames < max_frames:
            # Decode a frame.
            surface = nvDec.DecodeSingleSurface()
            if surface.Empty():
                break
            rgb_surface = conv.Execute(surface)
            frame = nvc.SurfaceToNumpy(rgb_surface)  # shape (H, W, 3)
            in_height, in_width, _ = frame.shape
            frame_gpu = cuda.mem_alloc(frame.nbytes)
            cuda.memcpy_htod(frame_gpu, frame)
            in_pitch = np.uint32(in_width * channels)

            # 1. Crop the frame.
            crop_out = np.empty((crop_height, crop_width, channels), dtype=np.uint8)
            crop_out_gpu = cuda.mem_alloc(crop_out.nbytes)
            grid_dim_crop = ((crop_width + block_dim[0] - 1) // block_dim[0],
                             ((crop_height + block_dim[1] - 1) // block_dim[1]))
            crop_kernel(crop_out_gpu, frame_gpu,
                        np.uint32(crop_width), np.uint32(crop_height),
                        in_pitch, np.uint32(channels),
                        np.uint32(crop_offsetX), np.uint32(crop_offsetY),
                        block=block_dim, grid=grid_dim_crop)

            # Save the cropped frame as a numpy file with a custom name.
            crop_host = np.empty((crop_height, crop_width, channels), dtype=np.uint8)
            cuda.memcpy_dtoh(crop_host, crop_out_gpu)
            crop_filename = "frame_{:04d}_cropped.npy".format(processed_frames)
            np.save(crop_filename, crop_host)
            print("Saved cropped frame as", crop_filename)

            # 2. Resize the cropped frame.
            resize_in_pitch = np.uint32(crop_width * channels)
            resize_out = np.empty((target_height, target_width, channels), dtype=np.uint8)
            resize_out_gpu = cuda.mem_alloc(resize_out.nbytes)
            grid_dim_resize = ((target_width + block_dim[0] - 1) // block_dim[0],
                               ((target_height + block_dim[1] - 1) // block_dim[1]))
            resize_kernel(resize_out_gpu, crop_out_gpu,
                          np.uint32(crop_width), np.uint32(crop_height),
                          resize_in_pitch,
                          np.uint32(target_width), np.uint32(target_height),
                          np.uint32(channels),
                          block=block_dim, grid=grid_dim_resize)

            # 3. Normalize the resized frame.
            norm_out = np.empty((target_height, target_width, channels), dtype=np.float32)
            norm_out_gpu = cuda.mem_alloc(norm_out.nbytes)  # float32: already 4 bytes per element.
            grid_dim_norm = ((target_width + block_dim[0] - 1) // block_dim[0],
                             ((target_height + block_dim[1] - 1) // block_dim[1]))
            normalize_kernel(norm_out_gpu, resize_out_gpu,
                             np.uint32(target_width), np.uint32(target_height),
                             np.uint32(target_width * channels),
                             np.uint32(channels),
                             scaleFactor,
                             mean_gpu,
                             block=block_dim, grid=grid_dim_norm)
            cuda.memcpy_dtoh(norm_out, norm_out_gpu)
            # The normalization kernel outputs in HWC order. Convert to CHW order.
            norm_out_chw = np.transpose(norm_out, (2, 0, 1))
            norm_filename = "frame_{:04d}_normalized.npy".format(processed_frames)
            np.save(norm_filename, norm_out_chw)
            print("Saved normalized frame as", norm_filename)

            processed_frames += 1

        mean_gpu.free()

if __name__ == '__main__':
    unittest.main()