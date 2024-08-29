#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>

//#define DATASET "/home/wellington/luna_files/luna_k5_f20_50000.dat"
#define DATASET "Effectiveness/Synth-7/luna_k7_f20_7000s.dat"
#define MAX_CLUSTERS 7
#define NF 20
#define MAX_POINTS 7000

void printitems(double *d_point, int p, int q, int natrib) {
    for (int i = 0; i < natrib; i++) {
        printf("e%d: %.2f, ", i, d_point[p + i]);
    }
    for (int j = 0; j < natrib; j++) {
        printf("f%d: %.2f, ", j, d_point[q + j]);
    }
}

double distance(double *d_point, int p, int q, int natrib) {
    double soma = 0;
    for (int i = 0; i < natrib; i++) {
        soma += pow((d_point[p + i] - d_point[q + i]), 2);
    }
    return sqrt(soma);
}

void min_intercluster_distance(int num_clusters, int *d_cluster_size, double *d_point, int *d_cluster_start, int natrib, double *d_r2) {
    double min_dist[MAX_CLUSTERS];

    for (int i = 0; i < num_clusters; i++) {
        min_dist[i] = DBL_MAX;
    }

    for (int i = 0; i < num_clusters; i++) {
        int t1 = (d_cluster_start[i + 1]) * natrib;
        for (int j = i + 1; j < num_clusters; j++) {
            int t2 = d_cluster_start[j + 1] * natrib;
            for (int p = d_cluster_start[i] * natrib; p < t1; p += natrib) {
                for (int q = d_cluster_start[j] * natrib; q < t2; q += natrib) {
                    double dist = distance(d_point, p, q, natrib);
                    if (dist < min_dist[i]) {
                        min_dist[i] = dist;
                    }
                }
            }
        }
    }

    double min = min_dist[0];
    for (int k = 1; k < num_clusters; k++) {
        if (min_dist[k] < min) {
            min = min_dist[k];
        }
    }
    *d_r2 = min;
}

void max_intracluster_distance(int num_clusters, int *d_cluster_size, double *d_point, int *d_cluster_start, int natrib, double *d_r1) {
    double max_dist[MAX_CLUSTERS];

    for (int i = 0; i < num_clusters; i++) {
        max_dist[i] = 0;
    }

    for (int i = 0; i < num_clusters; i++) {
        int t1 = (d_cluster_start[i + 1]) * natrib - natrib;
        int t2 = d_cluster_start[i + 1] * natrib;
        for (int p = d_cluster_start[i] * natrib; p < t1; p += natrib) {
            for (int q = p + natrib; q < t2; q += natrib) {
                double dist = distance(d_point, p, q, natrib);
                if (dist > max_dist[i]) {
                    max_dist[i] = dist;
                }
            }
        }
    }

    double max = max_dist[0];
    for (int k = 1; k < num_clusters; k++) {
        if (max_dist[k] > max) {
            max = max_dist[k];
        }
    }
    *d_r1 = max;
}

int main() {
    int num_clusters;
    int cluster_size[MAX_CLUSTERS];
    //point[MAX_POINTS][NF];
    static float point[MAX_POINTS][NF];
    //double **point = (double **)malloc(MAX_POINTS * sizeof(double *));
    int cluster_start[MAX_CLUSTERS + 1];
    int size = 0;
    int natrib;
    clock_t inicio, fim;
    double tempo_gasto;
    double h_r1, h_r2;

    /*for (int i = 0; i < MAX_POINTS; i++) {
        point[i] = (double *)malloc(NF * sizeof(double));
        if (!point[i]) {
            fprintf(stderr, "Memory allocation failed for 'point[%d]'.\n", i);
            exit(EXIT_FAILURE);
        }
    }*/

    FILE *fp;
    //fp = fopen("/content/drive/MyDrive/Faculdade./TCC./VDU./Aplicando CUDA no VDU./Datasets./Iris/iris_k3_f4_150.dat", "r");
    //fp = fopen("/content/drive/MyDrive/Faculdade./TCC./VDU./Aplicando CUDA no VDU./Datasets./water_k2_f9_3276.dat", "r");
    //fp = fopen("/content/drive/MyDrive/Faculdade./TCC./VDU./Aplicando CUDA no VDU./Datasets./Mozila_k2_f5_15545.dat", "r");
    //fp = fopen("/content/drive/MyDrive/Faculdade./TCC./VDU./Aplicando CUDA no VDU./Datasets./Letter./Letter_k26_f16_20000.dat", "r");
    //fp = fopen("/content/drive/MyDrive/Faculdade./TCC./VDU./Aplicando CUDA no VDU./Algoritmo LUNA./Data/luna_k5_f20_2500000.dat", "r");
    fp = fopen(DATASET, "r");


    fscanf(fp, "%d %d", &num_clusters, &natrib);
    num_clusters = MAX_CLUSTERS;

    for (int k = 0; k < num_clusters; k++) {
        fscanf(fp, "%d", &cluster_size[k]);
        size += cluster_size[k];
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < natrib; j++) {
            fscanf(fp, "%f", &point[i][j]);
        }
    }
    fclose(fp);

    cluster_start[0] = 0;
    for (int i = 1; i < num_clusters + 1; i++) {
        cluster_start[i] = cluster_start[i - 1] + cluster_size[i - 1];
    }

    double *d_r1, *d_r2;
    int *d_cluster_size, *d_cluster_start;
    double *d_point;

    d_r1 = (double *)malloc(sizeof(double));
    d_r2 = (double *)malloc(sizeof(double));
    d_cluster_size = (int *)malloc(MAX_CLUSTERS * sizeof(int));
    d_cluster_start = (int *)malloc((MAX_CLUSTERS + 1) * sizeof(int));
    d_point = (double *)malloc(MAX_POINTS * NF * sizeof(double));

    inicio = clock();

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < natrib; j++) {
            d_point[i * natrib + j] = point[i][j];
        }
    }

    for (int i = 0; i < num_clusters; i++) {
        d_cluster_size[i] = cluster_size[i];
    }

    for (int i = 0; i < num_clusters + 1; i++) {
        d_cluster_start[i] = cluster_start[i];
    }

    max_intracluster_distance(num_clusters, d_cluster_size, d_point, d_cluster_start, natrib, d_r1);

    min_intercluster_distance(num_clusters, d_cluster_size, d_point, d_cluster_start, natrib, d_r2);

    h_r1 = *d_r1;
    h_r2 = *d_r2;

    printf("Min intercluster distance: %.2f\n", h_r2);
    printf("Max intracluster distance: %.2f\n", h_r1);

    printf("The Dunn index is: %.9lf\n", h_r2 / h_r1);

    fim = clock();
    tempo_gasto = (double)(fim - inicio) / CLOCKS_PER_SEC;
    printf("Tempo gasto: %lf milissegundos\n", 1000.0 * tempo_gasto);

    /*for (int i = 0; i < MAX_POINTS; i++) {
        free(point[i]);
    }
    free(point);*/

    free(d_r1);
    free(d_r2);
    free(d_cluster_size);
    free(d_cluster_start);
    free(d_point);

    return 0;
}