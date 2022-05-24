#include "cuda.cuh"

#include <cstring>

#include "helper.h"

#include "device_launch_parameters.h"


///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of output image
Image cuda_output_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// image data size
size_t image_data_size;

// CUDA device
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
__device__ unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
__device__ unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
__device__ unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
__device__ unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
__device__ unsigned long long* d_global_pixel_sum;

// CUDA constant
__constant__ unsigned int d_CHANNELS;
__constant__ unsigned int d_TILES_X, d_TILES_Y;

unsigned long long* host_mosaic_sum;
unsigned char* host_mosaic_value;


__global__ void sum_tile(unsigned char const* __restrict__ input_image_data, unsigned long long* mosaic_sum) {

	/*
	* gridDim.x * blockIdx.y: block number in previous lines
	*/
	const unsigned int tile_index = (gridDim.x * blockIdx.y + blockIdx.x) * d_CHANNELS;
	const unsigned int tile_offset = (TILE_SIZE * TILE_SIZE * gridDim.x * blockIdx.y + TILE_SIZE * blockIdx.x) * d_CHANNELS;
	const unsigned int pixel_offset = (blockDim.x * gridDim.x * threadIdx.y + threadIdx.x) * d_CHANNELS;

	for (unsigned int ch = 0; ch < d_CHANNELS; ch++) {
		const unsigned char pixel = input_image_data[tile_offset + pixel_offset + ch];
		// sum up all the r/g/b channel in current tile
		atomicAdd(&mosaic_sum[tile_index + ch], pixel);
	}
}

__global__ void compact_mosaic(unsigned long long* mosaic_sum, unsigned char* mosaic_value, unsigned long long* global_pixel_sum) {
	/*
	* blockIdx.y * gridDim.x * blockDim.x * blockDim.y: Calculate excuted threads based on block y index (multiply whole block size in a grid line)
	* blockIdx.x * blockDim.x * blockDim.y: Calculate fully excuted blocked in current running grid line
	* threadIdx.y * blockDim.x + threadIdx.x: excuted threads in a block
	*/
	unsigned int i = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int mosaic_sum_index = i * d_CHANNELS + threadIdx.z;

	mosaic_value[mosaic_sum_index] = (unsigned char)(mosaic_sum[mosaic_sum_index] / TILE_PIXELS);

	atomicAdd(&global_pixel_sum[threadIdx.z], mosaic_value[mosaic_sum_index]);
}

__global__ void broadcast(unsigned char const* __restrict__ mosaic_value, unsigned char* output_image_data) {
	const unsigned int tile_index = (gridDim.x * blockIdx.y + blockIdx.x) * d_CHANNELS;
	const unsigned int tile_offset = (TILE_SIZE * TILE_SIZE * gridDim.x * blockIdx.y + TILE_SIZE * blockIdx.x) * d_CHANNELS;
	const unsigned int pixel_offset = (blockDim.x * gridDim.x * threadIdx.y + threadIdx.x) * d_CHANNELS;

	for (unsigned int ch = 0; ch < d_CHANNELS; ch++) {
		*(output_image_data + tile_offset + pixel_offset + ch) = mosaic_value[tile_index + ch];
	}
}

void cuda_begin(const Image* input_image) {
	// These are suggested CUDA memory allocations that match the CPU implementation
	// If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

	cuda_TILES_X = input_image->width / TILE_SIZE;
	cuda_TILES_Y = input_image->height / TILE_SIZE;

	// Allocate buffer for calculating the sum of each tile mosaic
	CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

	// Allocate buffer for storing the output pixel value of each tile
	CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

	image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);

	// Allocate copy of input image
	cuda_input_image = *input_image;
	cuda_input_image.data = (unsigned char*)malloc(image_data_size);
	memcpy(cuda_input_image.data, input_image->data, image_data_size);

	// Allocate copy of input image
	cuda_output_image = *input_image;
	cuda_input_image.data = (unsigned char*)malloc(image_data_size);

	// Allocate and fill device buffer for storing input image data
	CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
	CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

	// Allocate and fill device buffer for storing input image channels number
	CUDA_CALL(cudaMemcpyToSymbol(d_CHANNELS, &input_image->channels, sizeof(unsigned int)));
	// Allocate and fill device buffer for storing tile x and y
	CUDA_CALL(cudaMemcpyToSymbol(d_TILES_X, &cuda_TILES_X, sizeof(unsigned int)));
	CUDA_CALL(cudaMemcpyToSymbol(d_TILES_Y, &cuda_TILES_Y, sizeof(unsigned int)));

	// Allocate device buffer for storing output image data
	CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

	// Allocate and zero buffer for calculation global pixel average
	CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));


	// allocate for skip dunctions
	// Allocate buffer for calculating the sum of each tile mosaic
	host_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long));
	// Allocate buffer for storing the output pixel value of each tile
	host_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char));
}

void cuda_stage1() {
	// Optionally during development call the skip function with the correct inputs to skip this stage
	// skip_tile_sum(&host_input_image, host_mosaic_sum);

	// init params for kernel
	const unsigned int block_width = (unsigned int)TILE_SIZE;
	// block per grid is equal to the tile_x and tile_y
	dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
	dim3 threadsPerBlock(block_width, block_width, 1);

	// init sum array by 0
	memset(host_mosaic_sum, 0, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
	CUDA_CALL(cudaMemset(d_mosaic_sum, 0, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long)));

	// Run CUDA
	sum_tile << < blocksPerGrid, threadsPerBlock >> > (d_input_image_data, d_mosaic_sum);
	cudaDeviceSynchronize();
	CUDA_CALL(cudaMemcpy(host_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

#ifdef VALIDATION
	// TODO: Uncomment and call the validation function with the correct inputs
	// You will need to copy the data back to host before passing to these functions
	// (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
	validate_tile_sum(&cuda_input_image, host_mosaic_sum);
#endif
}

void cuda_stage2(unsigned char* output_global_average) {
	// Optionally during development call the skip function with the correct inputs to skip this stage
	// skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, host_mosaic_sum, host_mosaic_value, output_global_average);

	unsigned long long* whole_image_sum = (unsigned long long*)malloc(cuda_input_image.channels * sizeof(unsigned long long));
	memset(whole_image_sum, 0, cuda_input_image.channels * sizeof(unsigned long long));

	CUDA_CALL(cudaMemcpy(d_mosaic_sum, host_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyHostToDevice));

	// blockIdx.z refer to the max channels supported
	dim3 threadsPerBlock(4, 4, 3);
	dim3 blocksPerGrid(cuda_TILES_X / 4, cuda_TILES_Y / 4, 1);

	compact_mosaic << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_sum, d_mosaic_value, d_global_pixel_sum);
	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(host_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(whole_image_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

	for (unsigned int ch = 0; ch < cuda_input_image.channels; ++ch) {
		output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
	}


#ifdef VALIDATION
	// TODO: Uncomment and call the validation functions with the correct inputs
	// You will need to copy the data back to host before passing to these functions
	// (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
	validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, host_mosaic_sum, host_mosaic_value, output_global_average);
#endif    
}
void cuda_stage3() {
	// Optionally during development call the skip function with the correct inputs to skip this stage
	// skip_broadcast(&cuda_input_image, host_mosaic_value, &host_output_image);

	// init params for kernel
	const unsigned int block_width = (unsigned int)TILE_SIZE;
	// block per grid is equal to the tile_x and tile_y
	dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
	dim3 threadsPerBlock(block_width, block_width, 1);

	broadcast << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_value, d_output_image_data);
	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(cuda_output_image.data, d_output_image_data, image_data_size, cudaMemcpyDeviceToHost));


#ifdef VALIDATION
	// TODO: Uncomment and call the validation function with the correct inputs
	// You will need to copy the data back to host before passing to these functions
	// (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
	validate_broadcast(&cuda_input_image, host_mosaic_value, &cuda_output_image);
#endif    
}
void cuda_end(Image* output_image) {
	// This function matches the provided cuda_begin(), you may change it if desired

	// Store return value
	output_image->width = cuda_input_image.width;
	output_image->height = cuda_input_image.height;
	output_image->channels = cuda_input_image.channels;
	memcpy(output_image->data, cuda_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));

	// Release allocations
	free(cuda_input_image.data);
	CUDA_CALL(cudaFree(d_mosaic_value));
	CUDA_CALL(cudaFree(d_mosaic_sum));
	CUDA_CALL(cudaFree(d_global_pixel_sum));
	CUDA_CALL(cudaFree(d_input_image_data));
	CUDA_CALL(cudaFree(d_output_image_data));
}
