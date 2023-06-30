# gdunn
GPU Dunn index calculation

Colab CUDA code - GPU_Dunn_Index.ipynb

Check these parameters

// MAX_POINTS => power of two closest to the dataset size

// NF => number of features of each point

// BLOCK_SIZE 128 keep this value

// MAX_BLOCKS => MAX_POINTS / BLOCK_SIZE

Example for the Synthetic dataset with 5000 points, 20 features each, 5 clusters

#define MAX_POINTS 5120

#define NF 20

#define MAX_CLUSTERS 5

#define MAX_BLOCKS 40

#define BLOCK_SIZE 128

Choose the input file:

// fp = fopen("test_k2_f2_10.dat", "r"); => 10 points, 2 features, 2 clusters

// fp = fopen("digits_k10_f64_1797.dat", "r"); => 1797 points, 64 features, 10 clusters

// fp = fopen("synth_k5_f20_5000s.dat", "r"); => 5000 points, 20 features, 5 clusters
