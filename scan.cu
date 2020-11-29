#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cycletimer.h"
extern float toBW(int bytes, float sec);
/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/*
upsweep phase of exclusive scan algorithm in parallel. 
*/
__global__ void upsweep_kernel(int *device_data, int length, int twod, int twod1){
    //compute overall index in 1D array device_data
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_step_index = index*twod1;
    if(two_step_index+twod1-1 < length){
        device_data[two_step_index+twod1-1] += device_data[two_step_index+twod-1]; 
    }
} 

__global__ void downsweep_kernel(int *device_data, int length, int twod, int twod1, int og_length){
    //compute overall index in 1D array device_data
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_step_index = index*twod1;
    int t = 0;
    int twod1_val = 0;
    if(two_step_index+twod1-1 < length){
        t = device_data[two_step_index+twod-1]; 
        twod1_val = device_data[two_step_index+twod1-1];
        device_data[two_step_index+twod-1] = twod1_val;
        device_data[two_step_index+twod1-1] += t; 
    }
} 


__global__ void set_extras_zeroes_kernel(int *device_data, int length, int og_length){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= og_length && index < length){
        device_data[index] = 0;
    }
}

__global__ void set_zeroes_kernel(int *device_data, int length){
    device_data[length-1] = 0;
}

void exclusive_scan(int* device_data, int length)
{
    const int threadsPerBlock = 512;
    int og_length = length;
    length = nextPow2(length);
    int blocks = (length+threadsPerBlock - 1) / threadsPerBlock;
    set_extras_zeroes_kernel<<<blocks, threadsPerBlock>>>(device_data, length, og_length);
    cudaThreadSynchronize();
    //upsweep phase
    for (int twod = 1; twod < length; twod*=2){
        int twod1 = twod*2;
        blocks = (length/twod + threadsPerBlock) / threadsPerBlock;       
        upsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, length, twod, twod1);
        cudaThreadSynchronize();
    }     
    const int zero_const = 0;
    cudaMemcpy(&device_data[length-1], &zero_const, sizeof(int),cudaMemcpyHostToDevice); 
    // downsweep phase
    for (int twod = length/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        blocks = (length/twod + threadsPerBlock) / threadsPerBlock;
        downsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, length, twod, twod1, og_length);
        cudaThreadSynchronize();
    }
 }

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    /*for(int i = 0; i < (end-inarray); i++){
        printf("Input i: %d, val: %d\n", i, inarray[i]);
    } 
    for(int i = 0; i < (end-inarray); i++){
        //if(resultarray[i]==0)
        printf("Final i: %d, val: %d\n", i, resultarray[i]);
    } */
    return overallDuration;
}

/*
 For each element in input array, check if it's peak
 If greater than both preceding and following => peak
 */
__global__ void peak_kernel(int *device_input, int length, int *device_output){
    //compute overall index in 1D array device_data
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(0 < index < length-1){
        int cur_elem = device_input[index];
        int prev_elem = device_input[index-1];
        int next_elem = device_input[index+1];
        if(cur_elem > prev_elem && cur_elem > next_elem){
            device_output[index] = 1;
        }
        else{
            device_output[index] = 0;
        }
    }
     
} 

__global__ void set_zeroes_peaks_kernel(int *device_input, int length){
    device_input[0] = 0;
    device_input[length-1] = 0;
}

__global__ void keep_peaks_indices_kernel(int *peaks_ind_arr, int *peaks_mask, int length){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length){
        if(peaks_mask[index] != 1){
            peaks_ind_arr[index] = -1; 
        }
        //printf("Peaks Index: %d, Val: %d\n", index, peaks_ind_arr[index]);
    }    
}

__global__ void place_peaks_indices_kernel(int *peaks_ind_arr, int *output, int length){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int cur;
    if(index < length){
        cur = peaks_ind_arr[index];
        if(cur != -1){
            output[cur] = index; 
        }
    }
}
__device__ int out_peaks_len = -1;

__global__ void find_output_end_kernel(int *device_output, int length){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int cur;
    int next;
    if(index < length){
        cur = device_output[index];
        if(index < length-1) next = device_output[index+1];
        else next = 0;
        //printf("Next: %d, Cur: %d\n", next, cur);
        if(next == -1 && cur != -1){
            //printf("out len: %d\n", index);
            //return index;
            out_peaks_len = index;
        } 
    }
    // return -1;
}

__global__ void set_negative(int *device_output, int length){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length){
        device_output[index] = -1;
    }
}
int find_peaks(int *device_input, int length, int *device_output) {
    const int threadsPerBlock = 512;
    const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    peak_kernel<<<blocks, threadsPerBlock>>>(device_input, length, device_output);
    cudaThreadSynchronize();
    //mark first & last indices as not peaks
    const int zero_const = 0;
    cudaMemcpy(&device_output[length-1], &zero_const, sizeof(int),cudaMemcpyHostToDevice); 
    cudaMemcpy(&device_output[0], &zero_const, sizeof(int),cudaMemcpyHostToDevice); 
    //keep a copy of peaks mask in device_input, which is needed no more
    cudaMemcpy(device_input, device_output, length*sizeof(int), cudaMemcpyDeviceToDevice);
    //perform exclusive prefix sum on the peaks mask, to get total peaks before 
    //a given index. this will be same as the index final output in which to fill
    //the identifying index of a peak 
    exclusive_scan(device_input, length);
    cudaThreadSynchronize();  
    //find last element in final output to return length
    int out_peaks_len_g;
    cudaMemcpy(&out_peaks_len_g, &device_input[length-1], sizeof(int), cudaMemcpyDeviceToHost); 
    //in device_input, with indices of peaks' indices final output array, we will
    //only keep the elements that are peaks. device_output still has peaks
    keep_peaks_indices_kernel<<<blocks, threadsPerBlock>>>(device_input, device_output, length);
    //eventually, place into device_output, at indices obtained from device_input,
    //the indices at which peaks were found (by checking device_input itself)  
    place_peaks_indices_kernel<<<blocks, threadsPerBlock>>>(device_input, device_output, length); 
    cudaThreadSynchronize();
    return out_peaks_len_g;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = currentSeconds();
    
    int result = find_peaks(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
