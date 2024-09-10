#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// define o nome do dataset, quantidade de clusters, features e instancias
#define DATASET "/home/wellington/luna_files/luna_k5_f20_2500000.dat"
#define NUM_CLUSTERS 5
#define NUM_FEATURES 20
#define MAX_INSTANCES 2500000

typedef struct {
    float features[NUM_FEATURES];
    int cluster;
} Instance;

typedef struct {
    float features[NUM_FEATURES];
} Centroid;


void calculate_centroids(Instance *instances, int num_instances, Centroid *centroids, int * cluster_size ) {
    memset(centroids, 0, NUM_CLUSTERS * sizeof(Centroid));
    int cluster_sizes[NUM_CLUSTERS] = {0};

    for (int i = 0; i < num_instances; i++) {
        int cluster = instances[i].cluster;
        for (int j = 0; j < NUM_FEATURES; j++) {
            centroids[cluster].features[j] += instances[i].features[j];
        }
        cluster_sizes[cluster]++;
    }
    
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            centroids[i].features[j] /= cluster_sizes[i];
        }
    }
}

void calculate_general_centroid_from_centroids(Centroid *centroids, Centroid *general_centroid, Instance * instances) {
    
    memset(general_centroid, 0, sizeof(Centroid));

    
    for (int i = 0; i < MAX_INSTANCES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            general_centroid->features[j] += instances[i].features[j];
        }
    }

    
    for (int j = 0; j < NUM_FEATURES; j++) {
        general_centroid->features[j] /= MAX_INSTANCES;
    }
}

float calculate_interMean(Centroid *centroids, Centroid *general_centroid) {
    float interMean = 0.0;

    for (int i = 0; i < NUM_CLUSTERS; i++) {
        float dist = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            float diff = centroids[i].features[j] - general_centroid->features[j];
            dist += diff * diff;
        }
        interMean += sqrt(dist);
    }

    interMean /= NUM_CLUSTERS;

    return interMean;
}

float calculate_intraMean(Instance *instances, int num_instances, Centroid *centroids) {
    float intraMean = 0.0;

    for (int i = 0; i < num_instances; i++) {
        Centroid cluster_centroid = centroids[instances[i].cluster];
        float dist = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            float diff = instances[i].features[j] - cluster_centroid.features[j];
            dist += diff * diff;
        }
        intraMean += sqrt(dist);
    }

    intraMean /= num_instances;

    return intraMean;
}

float calculate_BDSilhouette(float interMean, float intraMean) {
    float maxMean = interMean > intraMean ? interMean : intraMean;
    float BDSilhouette = (interMean - intraMean) / maxMean;
    return BDSilhouette;
}

void print_instance(Instance instance) {
    printf("Características: ");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%f ", instance.features[i]);
    }
    printf("\nCluster: %d\n", instance.cluster);
}

void print_centroids(Centroid *centroids) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Centróide do Cluster %d: ", i);
        for (int j = 0; j < NUM_FEATURES; j++) {
            printf("%f ", centroids[i].features[j]);
        }
        printf("\n");
    }
}

void print_centroid(Centroid centroid) {
    printf("Centróide: ");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%f ", centroid.features[i]);
    }
    printf("\n");
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    int clusters, features, cluster_size[NUM_CLUSTERS], size = 0, i;

    Instance* instances = (Instance*) malloc(MAX_INSTANCES * sizeof(Instance));
    Centroid centroids[NUM_CLUSTERS], general_centroid;
    
   
    FILE *file = fopen(DATASET, "r");
    if (file == NULL) {
        printf("Não foi possível abrir o arquivo %s\n", DATASET);
        return -1;
    }

    fscanf(file, "%d %d", &clusters, &features); 

    for (int k = 0; k < NUM_CLUSTERS; k++) {
        fscanf(file, "%d", &cluster_size[k]);
        size = size + cluster_size[k];
    }

    int k = 0;
    int atual = cluster_size[0];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            fscanf(file, "%f,", &instances[i].features[j]);
        }

        instances[i].cluster = k;
        if(atual-1 == i){
                k++;
                atual += cluster_size[k];
        }
    }

    fclose(file);

    if (size < 0) {
        printf("Erro ao ler o dataset\n");
        return -1;
    }
     start = clock();
    calculate_centroids(instances, size, centroids, cluster_size);
    
    calculate_general_centroid_from_centroids(centroids, &general_centroid, instances);
    
    print_centroid(general_centroid);
    print_centroids(centroids);

    float interMean = calculate_interMean(centroids, &general_centroid);
    

    float intraMean = calculate_intraMean(instances, size, centroids);
    

    float BDSilhouette = calculate_BDSilhouette(interMean, intraMean);
    
    end = clock();
    printf("InterMean: %f\n", interMean);
    printf("IntraMean: %f\n", intraMean);
    printf("BD-Silhouette: %f\n", BDSilhouette);
    
    free(instances);


    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Tempo de execução: %f segundos\n", cpu_time_used);

    return 0;
}