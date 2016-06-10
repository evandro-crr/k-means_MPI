#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define RANDNUM_W 521288629;
#define RANDNUM_Z 362436069;

unsigned int randum_w = RANDNUM_W;
unsigned int randum_z = RANDNUM_Z;

void srandnum(int seed) {
  unsigned int w, z;
  w = (seed * 104623) & 0xffffffff;
  randum_w = (w) ? w : RANDNUM_W;
  z = (seed * 48947) & 0xffffffff;
  randum_z = (z) ? z : RANDNUM_Z;
}

unsigned int randnum(void) {
  unsigned int u;
  randum_z = 36969 * (randum_z & 65535) + (randum_z >> 16);
  randum_w = 18000 * (randum_w & 65535) + (randum_w >> 16);
  u = (randum_z << 16) + randum_w;
  return (u);
}

typedef float* vector_t;

int npoints;
int dimension;
int ncentroids;
float mindistance;
int seed;
vector_t *data, *centroids;
int *map;
int *dirty;
int too_far;
int has_changed;
int size, rank;

float v_distance(vector_t a, vector_t b) {
  float distance = 0;
  for (int i = 0; i < dimension; i++)
    distance +=  pow(a[i] - b[i], 2);
  return sqrt(distance);
}

struct arg_in_populate {
  int begin;
  int end;
  vector_t *data;
  int *map;
  vector_t *centroids;
};

struct arg_out_populate {
  int *map;
  int *dirty;
  int too_far;
};

static struct arg_out_populate populate(struct arg_in_populate arg_in) {
  struct arg_out_populate arg_out;
  arg_out.too_far = 0;
  arg_out.dirty = calloc(ncentroids, sizeof(int));

  for (int i = arg_in.begin; i < arg_in.end; i++) {
    float distance = v_distance(centroids[arg_in.map[i]], arg_in.data[i]);
    /* Look for closest cluster. */
    for (int j = 0; j < ncentroids; j++) {
      /* Point is in this cluster. */
      if (j == arg_in.map[i]) continue;
      float tmp = v_distance(arg_in.centroids[j], arg_in.data[i]);
      if (tmp < distance) {
        arg_out.map[i] = j;
        distance = tmp;
        arg_out.dirty[j] = 1;
      }
    }
    /* Cluster is too far away. */
    if (distance > mindistance)
      arg_out.too_far = 1;
  }

  arg_out.map = arg_in.map;

  return arg_out;
}

struct arg_in_compute_centroids {
  int begin;
  int end;
  vector_t *data;
  int *dirty;
  int *map;
};

struct arg_out_compute_centroids {
  int has_changed;
  vector_t *centroids;
};

static struct arg_out_compute_centroids compute_centroids(struct arg_in_compute_centroids arg_in) {
  struct arg_out_compute_centroids arg_out;
  arg_out.has_changed = 0;
  arg_out.centroids = calloc(ncentroids, sizeof(vector_t));

  /* Compute means. */
  for (int i = arg_in.begin; i < arg_in.end; i++) {
    if (!arg_in.dirty[i]) continue;
    arg_out.centroids[i] = calloc(dimension, sizeof(float));
    /* Compute cluster's mean. */
    int population = 0;
    for (int j = 0; j < npoints; j++) {
      if (arg_in.map[j] != i) continue;
      for (int k = 0; k < dimension; k++)
        arg_out.centroids[i][k] += arg_in.data[j][k];
      population++;
    }
    if (population > 1) {
      for (int k = 0; k < dimension; k++)
        arg_out.centroids[i][k] *= 1.0/population;
    }
    arg_out.has_changed = 1;
  }

  return arg_out;
}

void master() {
  printf("Master: %d\n", rank);
  if (!(data = calloc(npoints, sizeof(vector_t))))
    exit(1);

  for (int i = 0; i < npoints; i++) {
    data[i] = calloc(dimension, sizeof(float));
    for (int j = 0; j < dimension; j++)
      data[i][j] = randnum() & 0xffff;
  }

  too_far = 0;
  has_changed = 0;

  if (!(centroids = calloc(ncentroids, sizeof(vector_t))))
    exit (1);
  if (!(map  = calloc(npoints, sizeof(int))))
    exit (1);
  if (!(dirty = calloc(ncentroids, sizeof(int))))
    exit (1);

  for (int i = 0; i < ncentroids; i++)
    centroids[i] = calloc(dimension, sizeof(float));
  for (int i = 0; i < npoints; i++)
    map[i] = -1;
  for (int i = 0; i < ncentroids; i++) {
    dirty[i] = 1;
    int j = randnum() % npoints;
    for (int k = 0; k < dimension; k++)
      centroids[i][k] = data[j][k];
    map[j] = i;
  }
  /* Map unmapped data points. */
  for (int i = 0; i < npoints; i++)
    if (map[i] < 0)
      map[i] = randnum() % ncentroids;

  do { /* Cluster data. */
    //populate();
    printf("populate bcast\n");
    for (int i = 0; i < npoints; ++i) {
      MPI_Bcast(data[i], dimension, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(map, npoints, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < ncentroids; ++i) {
      MPI_Bcast(centroids[i], dimension, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    for (int i = 1; i < size; ++i) {
      int *tmp_map;
      int *tmp_dirty;
      int *tmp_too_far;
      MPI_Recv(tmp_map, npoints, MPI_INT, i, 0, MPI_COMM_WORLD, NULL);
      MPI_Recv(tmp_dirty, ncentroids, MPI_INT, i, 1, MPI_COMM_WORLD, NULL);
      MPI_Recv(tmp_too_far, 1, MPI_INT, i, 2, MPI_COMM_WORLD, NULL);
      for (int j = 0; j < npoints; ++j) {
        map[j] = tmp_map[j];
      }
      for (int j = 0; j < ncentroids; ++j) {
        dirty[j] = tmp_dirty[j];
      }
      too_far = too_far | *tmp_too_far;
    }
    printf("populate end\n");
    //compute_centroids();
    printf("compute_centroids bcast\n");
    for (int i = 0; i < npoints; ++i) {
      MPI_Bcast(data[i], dimension, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(dirty, ncentroids, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(map, npoints, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 1; i < size; ++i) {
      int *tmp_has_changed;
      vector_t *tmp_centroids;
      MPI_Recv(tmp_has_changed, 1, MPI_INT, i, 3, MPI_COMM_WORLD, NULL);
      for (int j = 0; j < ncentroids; ++j) {
        MPI_Recv(tmp_centroids[j], dimension, MPI_FLOAT, i, 4, MPI_COMM_WORLD, NULL);
      }
      has_changed = *tmp_has_changed | has_changed;
      for (int j = 0; j < ncentroids; ++j) {
        for (int k = 0; k < dimension; ++k) {
          centroids[j][k] = tmp_centroids[j][k];
        }
      }
    }
    printf("compute_centroids end\n");
  } while (too_far && has_changed);

  for (int i = 0; i < ncentroids; i++) {
    printf("\nPartition %d:\n", i);
    for (int j = 0; j < npoints; j++)
      if(map[j] == i)
        printf("%d ", j);
  }
  printf("\n");

  for (int i = 0; i < ncentroids; i++)
    free(centroids[i]);
  free(centroids);
  free(dirty);
  free(map);
  for (int i = 0; i < npoints; i++)
    free(data[i]);
  free(data);
}

void slave() {
  printf("Slave: %d\n", rank);
  struct arg_in_populate arg_in_p;
  struct arg_out_populate arg_out_p;
  struct arg_in_compute_centroids arg_in_c;
  struct arg_out_compute_centroids arg_out_c;

  arg_in_p.data = calloc(npoints, sizeof(vector_t));
  for (int i = 0; i < npoints; i++)
    arg_in_p.data[i] = calloc(dimension, sizeof(float));
  arg_in_p.centroids = calloc(ncentroids, sizeof(vector_t));
  for (int i = 0; i < ncentroids; i++)
    arg_in_p.centroids[i] = calloc(dimension, sizeof(float));
  arg_in_p.map = calloc(npoints, sizeof(int));

  arg_in_c.data = calloc(npoints, sizeof(vector_t));
  for (int i = 0; i < npoints; i++)
    arg_in_c.data[i] = calloc(dimension, sizeof(float));

  arg_in_c.dirty = calloc(ncentroids, sizeof(int));

  arg_in_c.map = calloc(npoints, sizeof(int));



  while (1) {
    //populate();
    printf("populate slave begin: %d\n", rank);
    for (int i = 0; i < npoints; ++i) {
      MPI_Bcast(arg_in_p.data[i], dimension, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(arg_in_p.map, npoints, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < ncentroids; ++i) {
      MPI_Bcast(arg_in_p.centroids[i], dimension, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    arg_in_p.begin = ((float) npoints / (float) (size-1)) * rank-1;
    arg_in_p.end = ((float) npoints /(float) (size)) * rank;

    arg_out_p = populate(arg_in_p);

    MPI_Send(arg_out_p.map, npoints, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(arg_out_p.dirty, ncentroids, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(&arg_out_p.too_far, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

    free(arg_out_p.dirty);
    printf("populate slave end: %d\n", rank);

    //compute_centroids();
    printf("compute_centroids slave begin: %d\n", rank);

    for (int i = 0; i < npoints; ++i) {
      printf("p%d\n", rank);
      MPI_Bcast(arg_in_c.data[i], dimension, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(arg_in_c.dirty, ncentroids, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(arg_in_c.map, npoints, MPI_INT, 0, MPI_COMM_WORLD);

    arg_in_c.begin = ((float) ncentroids / (float) (size-1)) * rank-1;
    arg_in_c.end = ((float) ncentroids /(float) (size)) * rank;

    arg_out_c = compute_centroids(arg_in_c);

    MPI_Send(&arg_out_c.has_changed, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
    for (int j = 0; j < ncentroids; ++j) {
        MPI_Send(arg_out_c.centroids[j], dimension, MPI_FLOAT, 0, 4, MPI_COMM_WORLD);
    }

    free(arg_out_c.centroids);
    printf("compute_centroids slave end: %d\n", rank);
  }

}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (argc != 6) {
    printf("Usage: npoints dimension ncentroids mindistance seed\n");
    exit (1);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  npoints = atoi(argv[1]);
  dimension = atoi(argv[2]);
  ncentroids = atoi(argv[3]);
  mindistance = atoi(argv[4]);
  seed = atoi(argv[5]);

  srandnum(seed);

  if (rank == 0) {
    master();
  } else {
    slave();
  }

  MPI_Finalize();
  return (0);
}
