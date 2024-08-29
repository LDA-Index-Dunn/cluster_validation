// This is the CUDA program for Dunn index calculation
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

// Test_k2_f2_10, Iris_k3_f4_150 - Digits_k10_f64_1797 - Electricity_k2_f8_45311
// MAX_POINTS 2048 (others) or 65536 (Electric) or 524288 (500K), 3145728, 14680064
// NF 2 (Test) 4 (Iris) 64 (Digits) 8 (Electricity)
// BLOCK_SIZE 128 (others) 64 (Digits)
// MAX_BLOCKS 256 (others) 1024 (Luna500K) 20048, 131072
// barcrawl_k10_f3_14057567 => 14680064 => 114688
#define MAX_POINTS 2621440
#define NF 20
#define BLOCK_SIZE 128

#define MAX_CLUSTERS 16
#define MAX_BLOCKS 20480


// Calculate the euclidian distance between two points in the CPU
float distance(float v1[], float v2[]) {
    float sum = 0.0;
    for (int d = 0; d < NF; d++) {
        sum += pow((v1[d] - v2[d]), 2);
    }
    return sqrt(sum);
}

// Calculate the euclidian distance between two points in the GPU
__device__ float distance(float *s_point, int p, int q, int nfeat) {
    float sum = 0.0;
    for (int d = 0; d < nfeat; d++) {
        sum = sum + pow((s_point[p+d] - s_point[q+d]), 2);
    }
    return sqrt(sum);
}

// Kernel that calculates the parcial sums (in each dimenstion) of the instances coordenates
__global__ void centroids(int cluster, float *d_centroid_tmp, float *d_point, int *d_cluster_start, int nfeat) {

  __shared__ float s_centroid[BLOCK_SIZE * NF];
  int tid = threadIdx.x;
  int lower = d_cluster_start[cluster];
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int size = d_cluster_start[cluster+1] - d_cluster_start[cluster];
  int p; //reduction step: 64, 32, 16, 8, 4, 2,1
    
  // All threads initialize with zero the shared memory
  for (int d = 0; d < nfeat; d++) {
    s_centroid[tid * nfeat + d] = 0.0;
  }
  __syncthreads();

  // Copy points from global memory to shared memory
  if (i < size) {
    for (int d = 0; d < nfeat; d++) {
      s_centroid[tid * nfeat + d] = (float ) d_point[(lower + i)*nfeat + d];
    }
  }
  __syncthreads();

  // Perform a local reduction on the memory shared data
  // It starts with 64 threads, then 32, 16, 8, 4, 2, 1
  p = blockDim.x / 2;
  while (p != 0) {
    if (tid < p) {
	    for (int d = 0; d < nfeat; d++) {
        s_centroid[tid*nfeat+d] = s_centroid[tid*nfeat+d] + s_centroid[(tid+p)*nfeat+d];
      }
    }
    __syncthreads();
    p = p/2;
  }

  // Thread zero of each block moves the local result to the global memory
  if (tid == 0) {
    for (int d = 0; d < nfeat; d++) {
        d_centroid_tmp[blockIdx.x * nfeat + d] = (float )s_centroid[d];
    }
  }
}

// Kernel that finds the cluster/point with the greatest distance from the centroid
//   maxs_intra<<<nblocks, BLOCK_SIZE>>>( cluster, d_index_tmp, d_maxs_tmp, d_centroid, d_point, d_cluster_start, nfeat );

__global__ void maxs_intra(int cluster, int *d_index_tmp, float *d_maxs_tmp, float *d_centroid, float *d_point, int *d_cluster_start, int nfeat) {
  
  __shared__ float s_point[(BLOCK_SIZE+1)*NF];
  __shared__ int s_pos[(BLOCK_SIZE+1)];
  __shared__ float s_dist[(BLOCK_SIZE+1)];
  int tid = threadIdx.x;
  int lower = d_cluster_start[cluster];
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int size = d_cluster_start[cluster+1] - d_cluster_start[cluster];
  int nb, d, p, r, q;
  float dist;
  
  // All threads initialize shared memory
  // s_pos[tid] = lower + tid;
  s_pos[tid] = lower + i;
  s_dist[tid] = 0.0;
  for (d = 0; d < nfeat; d++) {
    s_point[tid*nfeat+d] = 0.0;
  }
  __syncthreads();

  // printf("\nNBlocks %d", gridDim.x);

  // Copy data points from global memory to shared memory
  if (i < size) {
    for (d = 0; d < nfeat; d++) {
      s_point[tid*nfeat+d] = d_point[(lower+i)*nfeat+d];
    }
  }
  __syncthreads();

  // store centroid in the last position of the vector shared memory to save memory
  if (tid == 0) {
    for (d = 0; d < nfeat; d++) {
      s_point[blockDim.x*nfeat+d] = d_centroid[cluster*nfeat+d]; 
    }       
  }
  __syncthreads();

  // adjust limit for the last block
  if (blockIdx.x == (gridDim.x -1)) {
    nb = size % blockDim.x;
  } else { 
    nb = blockDim.x; 
  }

  // each thread calculates dist
  if (tid < nb) {
    r = tid*nfeat; // point index
    q = blockDim.x*nfeat; // centroid index
    dist = distance(s_point, r, q, nfeat);
    // s_point[tid*nfeat] = dist;
    s_dist[tid] = dist;
  }
  __syncthreads();

  // reduction to find the maximum distance
  /* p = blockDim.x / 2; // log steps
  while (p != 0) {
    if (tid < p) {
      if (s_point[tid*nfeat] < s_point[(tid+p)*nfeat]) {
        s_point[tid*nfeat] = s_point[(tid+p)*nfeat];
        s_pos[tid] = s_pos[tid+p];  
      }
    }
    __syncthreads();
    p = p/2;
  } */

  p = blockDim.x / 2; // log steps
  while (p != 0) {
    if (tid < p) {
      if (s_dist[tid] < s_dist[tid+p]) {
        s_dist[tid] = s_dist[tid+p];
        s_pos[tid] = s_pos[tid+p];  
      }
    }
    __syncthreads();
    p = p/2;
  }

  // Thread zero of each block copy data to glocal memory
  if (tid == 0) {
    d_index_tmp[blockIdx.x] = s_pos[0];
    // d_maxs_tmp[blockIdx.x] = s_point[0];
    d_maxs_tmp[blockIdx.x] = s_dist[0];
  }
}

int main()
{
  int num_clusters; // number of clusters
  static int cluster_size[MAX_CLUSTERS]; //cluster sizes
  static float point[MAX_POINTS][NF]; // cluster data
  float *d_point; // GPU cluster data
  static float centroid[MAX_CLUSTERS+1][NF]; // centroid data
  float *d_centroid; // GPU centroid data
  static float centroid_tmp[MAX_BLOCKS][NF]; // centroid temporary data
  float *d_centroid_tmp; // GPU centroid temporary data
  static float centroid_global[NF]; // global centroid
  static int index_tmp[MAX_BLOCKS]; // index temporary data
  int *d_index_tmp; // GPU index temporary data
  static float maxs_tmp[MAX_BLOCKS]; // max values temporary
  float *d_maxs_tmp; // GPU max values temporary
  static int cluster_start[MAX_CLUSTERS+1]; // start cluster indexes
  int *d_cluster_start; // GPU start cluster indexes
  FILE *fp; // file pointer
  int size = 0; // total number of points
  int nfeat; // number of attributes
  clock_t start, stop; // measure time
  double running_time; // running time
  int nblocks; // number of blocks
  int cluster; // current cluster
  float sum; // sum of elements
  float dist; // distance
  double max_distance; // maximum distance
  float min_distance; // minimum distance
  int cluster1; // cluster chosen
  int p1; // index chosen

  // Input the number of clusters and the cluster information
  // Format: 1st line: #clusters #features, 2nd: cluster sizes, 3rd: data
 
  // fp = fopen("test_k2_f2_10.dat", "r");
  // fp = fopen("iris_k3_f4_150.dat", "r");
  // fp = fopen("digits_k12_f64_1797s.dat", "r");
  // fp = fopen("electricity_k2_f8_45311.dat", "r");
  // fp = fopen("iris_k3_f4_150.dat", "r");
  // fp = fopen("digits_k13_f64_1797s.dat", "r");
  // fp = fopen("luna_k9_f20_7000s.dat", "r");
  // fp = fopen("satimage_k8_f36_6430s.dat", "r");
  // fp = fopen("aggregation_k9_f2_788s.dat", "r");
   fp = fopen("/home/wellington/luna_files/luna_k5_f20_2500000.dat", "r");
  // fp = fopen("texture_k13_f40_5500s.dat", "r");
  // fp = fopen("barcrawl_k10_f3_14057567.dat", "r");

  // Read file (upload file first if running in Collab)
  fscanf(fp, "%d %d", &num_clusters, &nfeat);
  num_clusters = 8;
  for (int k = 0; k < num_clusters; k++) {
    fscanf(fp, "%d", &cluster_size[k]);
    size = size + cluster_size[k];
    //printf("\ncluster_size %d", cluster_size[k]);
  }
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < nfeat; j++) {
       // fscanf(fp, "%f", &point[i][j]);
       if ( fscanf(fp, "%f", &point[i][j]) == 1) {
         fscanf(fp, ",");
       }
    }
  } 
  fclose(fp);

  // prefix sum to find out the beginning of each cluster
  cluster_start[0] = 0;
  for (int i = 1; i < num_clusters+1; i++) {
    cluster_start[i] = cluster_start[i-1] + cluster_size[i-1];
  }

  // Allocate GPU memory
  cudaMalloc(&d_cluster_start, (MAX_CLUSTERS+1)*sizeof(int));
  cudaMalloc(&d_point, MAX_POINTS*NF*sizeof(float));
  cudaMalloc(&d_centroid_tmp, MAX_BLOCKS*NF*sizeof(float));
  cudaMalloc(&d_index_tmp, MAX_BLOCKS*sizeof(int));
  cudaMalloc(&d_maxs_tmp, MAX_BLOCKS*sizeof(float));
  cudaMalloc(&d_centroid, MAX_CLUSTERS*NF*sizeof(float));

  // start clock to measure running time
  start = clock();

  // Copy data (cluster points and start indices) to the GPU
  cudaMemcpy(d_point, point, MAX_POINTS*nfeat*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cluster_start, cluster_start, (MAX_CLUSTERS+1)*sizeof(int), cudaMemcpyHostToDevice);

  // find centroids: launch the kernel for each cluster
  for (cluster = 0; cluster < num_clusters; cluster++) {
    
    // Number of blocks is size of cluster divided by the block size
    nblocks = (cluster_size[cluster] + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // launch kernel and verify if got any error
    centroids<<<nblocks, BLOCK_SIZE>>>( cluster, d_centroid_tmp, d_point, d_cluster_start, nfeat );
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(error)); exit(-1); }

    // Wait for the kernel to finish and copy centroid temporary data for the host (CPU)
    // The kernel returns the parcial sums of each block
    cudaDeviceSynchronize();
    cudaMemcpy(&centroid_tmp, d_centroid_tmp, MAX_BLOCKS*NF*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Calculate centroid and store it in the centroid_tmp
    // The parcial sums need to be accumulated and divided by the cluster size
    for (int i = 0; i < nfeat; i++) {
      sum = 0.0;
      for (int j = 0; j < nblocks; j++) {
        sum = sum + centroid_tmp[j][i];
      }
      centroid_global[i] = centroid_global[i] + sum;
      centroid_tmp[0][i] = sum / (float )cluster_size[cluster];
      centroid[cluster][i] = centroid_tmp[0][i];
    } 
  }

  //printf("\nCentroid Global: ");
  for (int i = 0; i < nfeat; i++) {
    centroid_global[i] = centroid_global[i] / size;
  }

  // Copy centroids to the GPU
  cudaMemcpy(d_centroid, centroid, MAX_CLUSTERS*NF*sizeof(float), cudaMemcpyHostToDevice);

  // Find the centroid closer to the global centroid
  min_distance = DBL_MAX;
  for (int i = 0; i < num_clusters; i++) {
    dist = distance(centroid[i], centroid_global);                               
    if (dist < min_distance) {
      min_distance = dist;
    }
  }
  // Now min_distance is the numerator of the Dunn index

  // Now, find maximum radius launching the kernel for each cluster again
  max_distance = 0.0;
  for (cluster = 0; cluster < num_clusters; cluster++) {
    
    // Number of blocks is size of cluster divided by the block size
    nblocks = (cluster_size[cluster] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // launch kernel and verify if got any error
    maxs_intra<<<nblocks, BLOCK_SIZE>>>( cluster, d_index_tmp, d_maxs_tmp, d_centroid, d_point, d_cluster_start, nfeat );
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(error)); exit(-1); }

    // Wait for the kernel to finish and copy maximum temporary data for the host (CPU)
    // The kernel returns the several maximums, one for each block
    cudaDeviceSynchronize();
    cudaMemcpy(&maxs_tmp, d_maxs_tmp, MAX_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&index_tmp, d_index_tmp, MAX_BLOCKS*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Calculate the global maximum and store it in the max_distance
    // The parcial maximums need to be compared and the global maximum stored
    // The cluster and the maximum point position need to be saved (cluster1 and p1)
    for (int j = 0; j < nblocks; j++) {
      if (maxs_tmp[j] > max_distance) {
        max_distance = maxs_tmp[j];
        p1 = index_tmp[j];
        cluster1 = cluster;
      }
    }
  }

  printf("\nCluster1: %d, p1: %d, max_distance: %.2f", cluster1, p1, max_distance);
/*
  // Now that we know the cluster (cluster1) with the point (p1) furthest to the 
  // centroid, we can calculate de diameter as the maximum distance between p1 
  // and another point in the same cluster
 
  // store p1 in the centroid[cluster1] vector, to re-use space, and move it to the GPU
  for (int d = 0; d < nfeat; d++) {
    // centroid[0][d] = point[p1][d];
    centroid[cluster1][d] = point[p1][d];
  } 
  cudaMemcpy(d_centroid, centroid, MAX_CLUSTERS*NF*sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  
  // Find maximum diameter in the right cluster (cluster1)
  // The point p1 is compared to all points of cluster1
  // This is done lauching the max_intra kernel once more
  cluster = cluster1;
  nblocks = (cluster_size[cluster] + (BLOCK_SIZE - 1)) / (BLOCK_SIZE);

  maxs_intra<<<nblocks, BLOCK_SIZE>>>( cluster, d_index_tmp, d_maxs_tmp, d_centroid, d_point, d_cluster_start, nfeat );
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)  { printf("CUDA error: %s\n", cudaGetErrorString(error)); exit(-1); }

  // Wait for the kernel to finish and copy maximum temporary data for the host (CPU)
  // The kernel returns the several maximums, one for each block
  cudaDeviceSynchronize();
  cudaMemcpy(&maxs_tmp, d_maxs_tmp, MAX_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&index_tmp, d_index_tmp, MAX_BLOCKS*sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Find out the maximum distance (one per block)
  // This is the denominator of the Dunn index
  max_distance = 0.0;
  for (int j = 0; j < nblocks; j++) {
    if (maxs_tmp[j] > max_distance) {
      max_distance = maxs_tmp[j];
    }
  }
*/
  // finalize runtime calculation
  stop = clock();
   
  // Print results
  printf("\nMin intercluster %.2f", min_distance);
  printf("\nMax intracluster %.2f", max_distance);
  printf("\nThe Dunn index: %.4f", min_distance / max_distance);
  
  // Print the time taken
  running_time = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("\nTime taken: %lf milissegundos\n", 1000.0*running_time);

  // Free GPU memory
  cudaFree( d_cluster_start );
  cudaFree( d_point );
  cudaFree( d_centroid_tmp );
  cudaFree( d_index_tmp );
  cudaFree( d_maxs_tmp );
  cudaFree( d_centroid );

  return 0;
}
