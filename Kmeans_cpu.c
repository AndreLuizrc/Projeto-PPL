/* ---- Tempo de execucao sequencial --------
    real    0m24.137s
    user    0m0.015s
    sys     0m0.062s
*/


#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_set_num_threads(n) ((void)0)
#endif

/* PARALLELIZACAO: tag para definir o numero de threads usadas pelo OpenMP.
   Pode ser alterada aqui ou sobrescrita na compilacao, por exemplo:
   gcc -fopenmp -DNUM_THREADS=8 Kmeans_cpu.c -o Kmeans_cpu */
#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

static int getThreadCount()
{
    return NUM_THREADS > 0 ? NUM_THREADS : 1;
}

typedef struct observation
{
    double x;  /**< abscissa of 2D data point */
    double y;  /**< ordinate of 2D data point */
    int group; /**< the group no in which this observation would go */
} observation;


typedef struct cluster
{
    double x;     /**< abscissa centroid of this cluster */
    double y;     /**< ordinate of centroid of this cluster */
    size_t count; /**< count of observations present in this cluster */
} cluster;


int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calculate Squared Distance*/
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

void calculateCentroid(observation observations[], size_t size,
                       cluster* centroid)
{
    double sumX = 0;
    double sumY = 0;
    int numThreads = getThreadCount();

    centroid->count = size;

    /* PARALLELIZACAO: reducao soma x/y em paralelo e atualiza cada ponto
       independentemente, evitando disputa por memoria compartilhada. */
#pragma omp parallel for reduction(+ : sumX, sumY) num_threads(numThreads) schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        sumX += observations[i].x;
        sumY += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x = sumX / centroid->count;
    centroid->y = sumY / centroid->count;
}

cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        /*
        If we have to cluster them only in one group
        then calculate centroid of observations and
        that will be a ingle cluster
        */
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if ((size_t)k < size)
    {
        int numThreads = getThreadCount();
        size_t partialLen = (size_t)numThreads * (size_t)k;
        double* partialX = (double*)calloc(partialLen, sizeof(double));
        double* partialY = (double*)calloc(partialLen, sizeof(double));
        size_t* partialCount = (size_t*)calloc(partialLen, sizeof(size_t));

        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000;  // Do until 99.99 percent points are in correct cluster
        do
        {
            memset(partialX, 0, partialLen * sizeof(double));
            memset(partialY, 0, partialLen * sizeof(double));
            memset(partialCount, 0, partialLen * sizeof(size_t));

            /* Initialize clusters */
#pragma omp parallel for num_threads(numThreads) schedule(static)
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }

            /* STEP 2*/
            /* PARALLELIZACAO: cada thread acumula em um vetor parcial proprio,
               evitando corrida ao somar x/y/count dos clusters compartilhados. */
#pragma omp parallel num_threads(numThreads)
            {
                int threadId = omp_get_thread_num();
                size_t base = (size_t)threadId * (size_t)k;

#pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;
                    partialX[base + group] += observations[j].x;
                    partialY[base + group] += observations[j].y;
                    partialCount[base + group]++;
                }
            }

            /* PARALLELIZACAO: combina as somas parciais de cada thread por
               cluster; como cada iteracao escreve em um cluster, nao ha corrida. */
#pragma omp parallel for num_threads(numThreads) schedule(static)
            for (int i = 0; i < k; i++)
            {
                double sumX = 0;
                double sumY = 0;
                size_t count = 0;

                for (int threadId = 0; threadId < numThreads; threadId++)
                {
                    size_t idx = (size_t)threadId * (size_t)k + (size_t)i;
                    sumX += partialX[idx];
                    sumY += partialY[idx];
                    count += partialCount[idx];
                }

                clusters[i].x = sumX;
                clusters[i].y = sumY;
                clusters[i].count = count;

                if (clusters[i].count > 0)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }

            /* STEP 3 and 4 */
            changed = 0;  // this variable stores change in clustering
            /* PARALLELIZACAO: cada ponto e reclassificado de forma independente;
               reduction soma com seguranca quantos pontos mudaram de grupo. */
#pragma omp parallel for reduction(+ : changed) num_threads(numThreads) schedule(static)
            for (size_t j = 0; j < size; j++)
            {
                int nearest = calculateNearst(observations + j, clusters, k);
                if (nearest != observations[j].group)
                {
                    changed++;
                    observations[j].group = nearest;
                }
            }
        } while (changed > minAcceptedError);  // Keep on grouping until we have
                                               // got almost best clustering

        free(partialX);
        free(partialY);
        free(partialCount);
    }
    else
    {
        /* If no of clusters is more than observations
           each observation can be its own cluster
        */
        clusters = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* PARALLELIZACAO: caso trivial, cada iteracao escreve em indices
           independentes de clusters/observations. */
#pragma omp parallel for num_threads(getThreadCount()) schedule(static)
        for (size_t j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }
    return clusters;
}


void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    double scale = 0, cx = 0, cy = 0;
    double* colors = (double*)malloc(sizeof(double) * (k * 3));
    int i;
    size_t j;
    double kd = k * 1.0;
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x)
        {
            max_x = pts[j].x;
        }
        if (min_x > pts[j].x)
        {
            min_x = pts[j].x;
        }
        if (max_y < pts[j].y)
        {
            max_y = pts[j].y;
        }
        if (min_y > pts[j].y)
        {
            min_y = pts[j].y;
        }
    }
    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    };
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    printf("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10,
           H + 10);
    printf(
        "/l {rlineto} def /m {rmoveto} def\n"
        "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
        "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
        "	gsave 1 setgray fill grestore gsave 3 setlinewidth"
        " 1 setgray stroke grestore 0 setgray stroke }def\n");
    for (int i = 0; i < k; i++)
    {
        printf("%g %g %g setrgbcolor\n", *(colors + 3 * i),
               *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i)
            {
                continue;
            }
            printf("%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2,
                   (pts[j].y - cy) * scale + H / 2);
        }
        printf("\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2,
               (cent[i].y - cy) * scale + H / 2);
    }
    printf("\n%%%%EOF");

    // free accquired memory
    free(colors);
}


static void test()
{
    size_t size = 100000L;
    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 5;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    // Free the accquired memory
    free(observations);
    free(clusters);
}


void test2()
{
    size_t size = 1000000L;
    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 11;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    // Free the accquired memory
    free(observations);
    free(clusters);
}

/*!
 * This function calls the test
 * function
 */
int main()
{
    /* PARALLELIZACAO: aplica a tag NUM_THREADS ao runtime do OpenMP. */
    omp_set_num_threads(getThreadCount());
    srand(time(NULL));
    // test();
    test2(); 
    return 0;
}
