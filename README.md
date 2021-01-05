# Parallel Image Processing Toolkit

Image processing toolbox for several image segmentation algorithms including Otsu Binarization, Edge Detection, and K-Means. The sequential versions of the algorithms are implemented using C++, and the parallel versions were implemented in both OpenMP and CUDA. The goal of this project is to implement and analyze the scalability, scope, and benefit of performing image segmentation in parallel.

Image segmentation techniques can be broadly classified into three categories - thresholding, edge detection and clustering. In this project we have implemented one algorithm from each of these main categories, and parallelized them using CUDA and OpenMP. Different parallelization strategies are used for each of these algorithms in order to obtain the best speedups. While these algorithms have the same common goal of image segmentation, their respective details are very different, and each one thus requires carefully testing strategies to get the optimal solution.

For results of the different approaches, please refer to https://kmeshx.github.io/pdfs/418_CP__Copy_.pdf

## Otsu Binarization 
Otsu Binarization is a adaptive thresholding algorithm in which the algorithm assumes that the distribution of the image is bimodal(one mode for
the background and the second for the foreground). Histograms for the image pixel distribution are created, and the threshold that causes the maximum inter-class(one class consists of pixels less than the threshold, the other consists of values above the threshold) variance is selected. 

### OpenMP
Two approaches are tested : parallelization over images(perfect workload balance for the same image), and parallelization within the algorithm)
Different approaches are tested to minimize contention. The final operation is conducted by sequentially updating the values for each pixel(this exceeds the performance as compared to atomic operations, multi-locking, and lock-free parallel implementations), parallelizing the operation to find the max-threshold using a work-efficient scan operation and setting the final values. 

### CUDA
Similar to OpenMP, two approaches are conducted. CUDA streaming allows us to hide the latency of the memory transfer. After this, kernel based operations are conducted to find the pixel value distributions, and max-variance calculation(using parallel exclusive scan)

## Edge Detection
Another way to binarize an image is to color the edges as one color and the non-edges as another. To use this, we can implement a edge detection algorithm that automatically identifies the edges. A common way of doing this is by finding the gradient of each pixel and comparing it to the surrounding pixels. In particular, we use a SOBEL operator to binarize the image


### OpenMP
Different approaches are presented including directly using the 2D kernel, and separating it into two 1D kernels. Another approach presented is the use of chunking(wherein the cache-size num elements are calculated together). 

### CUDA
All three approaches presented above are also implemented in CUDA(with slight modifications to cater to the GPU architecture)

## K-Means
Here images are segmented into K colors using K-means based on pixel value distance

### OpenMP 
The assignments of points are conducted in parallel, and OpenMP reductions are used to sum over variables for each cluster. 

### CUDA
Different approaches are presented including the direct conversion of the above stated OpenMP algorithm in kernel-format, use of a global scan operation, and chunked updates(listed in increasing performance)


