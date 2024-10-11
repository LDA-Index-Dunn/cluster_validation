// This is the C program for Silhouette index calculation
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>

// Test_k2_f2_10, Iris_k3_f4_150 - Digits_k10_f64_1797 - Electricity_k2_f8_45311
// MAX_POINTS 2048 (others) or 65536 (Electric) or 524288 (500K), 3145728, 14680064
// NF 2 (Test) 4 (Iris) 64 (Digits) 8 (Electricity) 20 (Luna)
#define MAX_POINTS 500000
#define NF 20
#define MAX_CLUSTERS 16

float float_max(float num1, float num2){
    if(num1 > num2){
        return num1;
    }
    return num2;
}

float euclidean_distance(float *row, float *vector) {
  float distance = 0.0;
  for (int i = 0; i < NF; i++) {
    float difference = row[i] - vector[i];
    distance += difference * difference;
  }
  return sqrt(distance);
}

int main()
{
  int num_clusters; // number of clusters
  static int cluster_size[MAX_CLUSTERS]; //cluster sizes
  static float point[MAX_POINTS][NF]; // cluster data
  static float centroid[MAX_CLUSTERS][NF]; // centroid data
  static float centroid_global[NF]; // global centroid
  int cluster_start[MAX_CLUSTERS+1]; // start cluster indexes
  FILE *fp; // file pointer
  int size = 0; // total number of points
  int nfeat; // number of attributes
  clock_t start, stop; // measure time
  double running_time; // running time
  float dist; // distance
  static float sum[MAX_CLUSTERS][NF];
  float sil_index; // Silhouette index
  int i, j, d, cluster1, cluster2, csize1, csize2, tmp;
  int lower1, lower2; // first index of each cluster
  float min_distance; // mininum distance between clusters
  float avg_intra_dist; // average intra cluster distance
  float avg_inter_dist; // average inter cluster distance
  float avg_intra_tot; // average intra cluster total distance
  float avg_inter_tot; // average inter cluster total distance

  // Input the number of clusters and the cluster information
  // Format: 1st line: #clusters #features, 2nd: cluster sizes, 3rd: data

  // printf("\nThe Silhouette index:");

  // fp = fopen("test_k2_f2_10.dat", "r");
  // fp = fopen("iris_k3_f4_150s.dat", "r");
  // fp = fopen("digits_k10_f64_1797.dat", "r");
  // fp = fopen("electricity_k2_f8_45311.dat", "r");
  // fp = fopen("iris_k3_f4_150s.dat", "r");
  // fp = fopen("digits_k10_f64_1797s.dat", "r");
   //fp = fopen("Effectiveness/Synth-7/luna_k8_f20_7000s.dat", "r");
  // fp = fopen("satimage_k8_f36_6430s.dat", "r");
  // fp = fopen("aggregation_k9_f2_788s.dat", "r");
   //fp = fopen("/home/wellington/luna_files/luna_k5_f20_10000.dat", "r");
  // fp = fopen("texture_k13_f40_5500s.dat", "r");
  // fp = fopen("barcrawl_k10_f3_14057567.dat", "r");
   fp = fopen("/home/wellington/luna_files/luna_k5_f20_500000.dat", "r");
   //fp = fopen("/home/wellington/luna_files/luna_k5_f20_2500000.dat", "r");
   //fp = fopen("/home/wellington/electricity_k2_f8_45311.dat", "r");

  // Read file (upload file first if running in Collab)
  fscanf(fp, "%d %d", &num_clusters, &nfeat);
  num_clusters = 5;
  for (int k = 0; k < num_clusters; k++) {
    fscanf(fp, "%d", &cluster_size[k]);
    size = size + cluster_size[k];
  }
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < nfeat; j++) {
       fscanf(fp, "%f", &point[i][j]);
    }
  }
  fclose(fp);

  // prefix sum to find out the beginning of each cluster
  cluster_start[0] = 0;
  for (int i = 1; i < num_clusters+1; i++) {
    cluster_start[i] = cluster_start[i-1] + cluster_size[i-1];
  }

  // start clock to measure running time
  start = clock();

  // Find minimum average distance between clusters
  min_distance = DBL_MAX;
  for (cluster1 = 0; cluster1 < num_clusters; cluster1++) {
    lower1 = cluster_start[cluster1];
    csize1 = cluster_start[cluster1+1] - cluster_start[cluster1];
	  for (cluster2 = cluster1+1; cluster2 < num_clusters; cluster2++) {
      lower2 = cluster_start[cluster2];
      csize2 = cluster_start[cluster2+1] - cluster_start[cluster2];
      avg_inter_dist = 0.0;
      for (i = 0; i < csize1; i++) {
        for (j = 0; j < csize2; j++) {
          dist = euclidean_distance(point[lower1+i], point[lower2+j]);
		      avg_inter_dist = avg_inter_dist + dist;
        }
      }
      avg_inter_dist = avg_inter_dist / (csize1 * csize2);
      if (avg_inter_dist < min_distance) {
        min_distance = avg_inter_dist;
      }
    }
  }
  avg_inter_tot = min_distance;

  printf("\nAvg intercluster %.2f", avg_inter_tot);

  // Find average intra cluster distance
  avg_intra_tot = 0.0;
  for (cluster1 = 0; cluster1 < num_clusters; cluster1++) {
    avg_intra_dist = 0.0; // test
    lower1 = cluster_start[cluster1];
    csize1 = cluster_start[cluster1+1] - cluster_start[cluster1];
    for (i = 0; i < csize1-1; i++) {
      for (j = i+1; j < csize1; j++) {
        dist = euclidean_distance(point[lower1+i], point[lower1+j]);
        avg_intra_dist = avg_intra_dist + dist;
      }
    }
    avg_intra_dist = avg_intra_dist * 2;
    avg_intra_dist = avg_intra_dist / ((csize1-1)*csize1);
    avg_intra_tot = avg_intra_tot + avg_intra_dist;
  }
  avg_intra_tot = avg_intra_tot / num_clusters;

  printf("\nAvg intracluster %.2f", avg_intra_tot);

  // Calculate the Silhouette index:
  // BD-Silhouette = (inter-cluster - intra-cluster) / max{inter-cluster, intra-cluster}
  sil_index = (avg_inter_tot - avg_intra_tot) / float_max(avg_inter_tot, avg_intra_tot);

  printf("\nThe Silhouette index: %.2f", sil_index);

  // finalize runtime calculation
  stop = clock();

  // Print the time taken
  running_time = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("\nTime taken: %lf milissegundos\n", 1000.0*running_time);

  return 0;
}
