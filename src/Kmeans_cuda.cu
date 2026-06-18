/* ---- Versao CUDA --------
   Compilacao sugerida:
       nvcc -O2 Kmeans_cuda.cu -o Kmeans_cuda.exe

   A geracao dos pontos e a impressao EPS continuam na CPU. O ciclo principal
   do K-Means e executado na GPU:
       1. acumula soma/count por cluster em parciais por bloco CUDA;
       2. recalcula centroides;
       3. reclassifica pontos em paralelo e conta quantos mudaram.
*/

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <cfloat>         /* DBL_MAX, DBL_MIN */
#include <cmath>          /* M_PI, sin, cos */
#include <cstdio>         /* printf, fprintf */
#include <cstdlib>        /* rand, malloc */
#include <cstring>        /* memset */
#include <ctime>          /* time, clock */
#include <cuda_runtime.h>

#ifndef CUDA_BLOCK_SIZE
#define CUDA_BLOCK_SIZE 256
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED 0
#endif

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

/* Wrapper simples para checar erros da API CUDA.
   Sem isso, uma falha de alocacao, copia ou kernel poderia passar despercebida
   e o programa continuaria com resultados incorretos. */
static void checkCuda(cudaError_t result, const char *call, const char *file,
                      int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line,
                call, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

static double getWallTime()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

static void printBenchmark(size_t size, int k, double generatedAt,
                           double clusteredAt, double finishedAt,
                           double startedAt)
{
    fprintf(stderr, "\nBenchmark CUDA: size=%zu k=%d block=%d\n", size, k,
            CUDA_BLOCK_SIZE);
    fprintf(stderr, "geracao_dados: %.6f s\n", generatedAt - startedAt);
    fprintf(stderr, "kmeans_cuda: %.6f s\n", clusteredAt - generatedAt);
    fprintf(stderr, "printEPS: %.6f s\n", finishedAt - clusteredAt);
    fprintf(stderr, "total_medido: %.6f s\n", finishedAt - startedAt);
}

__global__ void clearAccumulatorsKernel(double *sumX, double *sumY,
                                        unsigned long long *count,
                                        size_t len)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        sumX[i] = 0.0;
        sumY[i] = 0.0;
        count[i] = 0;
    }
}

/* atomicAdd(double*) so existe nativamente em GPUs Pascal ou superiores
   quando o codigo e compilado para arquitetura >= sm_60.

   Como alguns nvcc compilam por padrao para uma arquitetura antiga, este helper
   usa o atomicAdd nativo quando possivel e cai para uma implementacao com
   atomicCAS quando necessario. Na sua RTX 4060, compilando com -arch=sm_89, o
   caminho usado sera o atomicAdd nativo. */
__device__ double atomicAddDouble(double *address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    unsigned long long int *addressAsUll =
        (unsigned long long int *)address;
    unsigned long long int old = *addressAsUll;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addressAsUll, assumed,
                        __double_as_longlong(value +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}

__global__ void accumulateBlockPartialsKernel(const observation *observations,
                                              size_t size, double *partialX,
                                              double *partialY,
                                              unsigned long long *partialCount,
                                              int k)
{
    /* Cada bloco CUDA acumula suas proprias somas parciais dos clusters.
       Exemplo: partialX[bloco * k + cluster].

       Isso reduz a disputa em atomics globais: em vez de todos os pontos da GPU
       atualizarem apenas k posicoes, cada bloco atualiza seu proprio conjunto de
       k posicoes. Depois outro kernel combina essas parciais. */
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size)
    {
        int group = observations[j].group;
        size_t idx = (size_t)blockIdx.x * k + group;
        atomicAddDouble(&partialX[idx], observations[j].x);
        atomicAddDouble(&partialY[idx], observations[j].y);
        atomicAdd(&partialCount[idx], 1ULL);
    }
}

__global__ void updateCentroidsKernel(cluster *clusters, const double *partialX,
                                      const double *partialY,
                                      const unsigned long long *partialCount,
                                      int k, int numPointBlocks)
{
    /* Um thread por cluster combina os acumuladores parciais de todos os blocos.
       Ao final, clusters[i] passa a guardar o novo centroide do cluster i. */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k)
    {
        double sumX = 0.0;
        double sumY = 0.0;
        unsigned long long count = 0;

        for (int block = 0; block < numPointBlocks; block++)
        {
            size_t idx = (size_t)block * k + i;
            sumX += partialX[idx];
            sumY += partialY[idx];
            count += partialCount[idx];
        }

        clusters[i].count = (size_t)count;
        if (count > 0)
        {
            clusters[i].x = sumX / (double)count;
            clusters[i].y = sumY / (double)count;
        }
    }
}

__device__ int calculateNearestDevice(const observation *o,
                                      const cluster *clusters, int k)
{
    double minD = DBL_MAX;
    int index = -1;

    for (int i = 0; i < k; i++)
    {
        double dx = clusters[i].x - o->x;
        double dy = clusters[i].y - o->y;
        double dist = dx * dx + dy * dy;
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }

    return index;
}

__global__ void reassignObservationsKernel(observation *observations,
                                           size_t size,
                                           const cluster *clusters, int k,
                                           unsigned long long *changed)
{
    /* Cada thread reclassifica uma observacao independente.
       Se o grupo mudou, incrementa changed com atomicAdd. Esse contador e usado
       no host para decidir se o K-Means ja convergiu o suficiente. */
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size)
    {
        int nearest = calculateNearestDevice(&observations[j], clusters, k);
        if (nearest != observations[j].group)
        {
            observations[j].group = nearest;
            atomicAdd(changed, 1ULL);
        }
    }
}

int calculateNearst(observation *o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
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
                       cluster *centroid)
{
    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

cluster *kMeansCuda(observation observations[], size_t size, int k)
{
    cluster *clusters = NULL;
    if (k <= 1)
    {
        clusters = (cluster *)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if ((size_t)k < size)
    {
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));

        /* A inicializacao aleatoria dos grupos ficou na CPU, como no codigo
           original. Ela ocorre uma unica vez e usa rand(), que tem estado global
           e nao e adequado para ser chamado diretamente por milhares de threads
           CUDA. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        observation *dObservations = NULL;
        cluster *dClusters = NULL;
        double *dPartialX = NULL;
        double *dPartialY = NULL;
        unsigned long long *dPartialCount = NULL;
        unsigned long long *dChanged = NULL;
        unsigned long long changed = 0;
        size_t minAcceptedError = size / 10000;

        /* Aloca na GPU os pontos, os centroides, os acumuladores parciais e o
           contador de mudancas. O prefixo d indica device, ou seja, memoria da
           GPU. */
        CUDA_CHECK(cudaMalloc((void **)&dObservations,
                              sizeof(observation) * size));
        CUDA_CHECK(cudaMalloc((void **)&dClusters, sizeof(cluster) * k));

        /* pointBlocks e a quantidade de blocos CUDA necessaria para cobrir os
           1.000.000 pontos. partialLen reserva k acumuladores para cada bloco. */
        int pointBlocks = (int)((size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        int clusterBlocks = (k + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        size_t partialLen = (size_t)pointBlocks * k;
        int partialClearBlocks =
            (int)((partialLen + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

        CUDA_CHECK(cudaMalloc((void **)&dPartialX,
                              sizeof(double) * partialLen));
        CUDA_CHECK(cudaMalloc((void **)&dPartialY,
                              sizeof(double) * partialLen));
        CUDA_CHECK(cudaMalloc((void **)&dPartialCount,
                              sizeof(unsigned long long) * partialLen));
        CUDA_CHECK(cudaMalloc((void **)&dChanged, sizeof(unsigned long long)));

        /* Copia os dados iniciais da CPU para a GPU. Depois disso, o loop
           principal do K-Means roda sem trazer todos os pontos de volta a cada
           iteracao; apenas o contador changed volta para a CPU. */
        CUDA_CHECK(cudaMemcpy(dObservations, observations,
                              sizeof(observation) * size,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dClusters, clusters, sizeof(cluster) * k,
                              cudaMemcpyHostToDevice));

        do
        {
            /* Cada iteracao refaz os centroides a partir da classificacao atual,
               depois reclassifica os pontos usando esses centroides. */
            CUDA_CHECK(cudaMemset(dChanged, 0, sizeof(unsigned long long)));

            /* Zera os acumuladores parciais antes de somar os pontos da
               iteracao atual. */
            clearAccumulatorsKernel<<<partialClearBlocks, CUDA_BLOCK_SIZE>>>(
                dPartialX, dPartialY, dPartialCount, partialLen);
            CUDA_CHECK(cudaGetLastError());

            /* Soma x, y e count de cada ponto no acumulador parcial do seu
               bloco CUDA e do seu cluster atual. */
            accumulateBlockPartialsKernel<<<pointBlocks, CUDA_BLOCK_SIZE>>>(
                dObservations, size, dPartialX, dPartialY, dPartialCount, k);
            CUDA_CHECK(cudaGetLastError());

            /* Combina as parciais de todos os blocos e calcula os novos
               centroides. */
            updateCentroidsKernel<<<clusterBlocks, CUDA_BLOCK_SIZE>>>(
                dClusters, dPartialX, dPartialY, dPartialCount, k,
                pointBlocks);
            CUDA_CHECK(cudaGetLastError());

            /* Para cada ponto, encontra o centroide mais proximo e atualiza o
               grupo. O kernel tambem conta quantos pontos mudaram. */
            reassignObservationsKernel<<<pointBlocks, CUDA_BLOCK_SIZE>>>(
                dObservations, size, dClusters, k, dChanged);
            CUDA_CHECK(cudaGetLastError());

            /* Copia apenas o contador de mudancas para a CPU. Esse valor decide
               se o loop continua, reproduzindo o criterio do codigo original. */
            CUDA_CHECK(cudaMemcpy(&changed, dChanged,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
        } while (changed > minAcceptedError);

        /* Ao final da convergencia, copia de volta os pontos com seus grupos
           finais e os centroides finais para que printEPS consiga gerar a imagem
           na CPU. */
        CUDA_CHECK(cudaMemcpy(observations, dObservations,
                              sizeof(observation) * size,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(clusters, dClusters, sizeof(cluster) * k,
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dObservations));
        CUDA_CHECK(cudaFree(dClusters));
        CUDA_CHECK(cudaFree(dPartialX));
        CUDA_CHECK(cudaFree(dPartialY));
        CUDA_CHECK(cudaFree(dPartialCount));
        CUDA_CHECK(cudaFree(dChanged));
    }
    else
    {
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (size_t j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = (int)j;
        }
    }
    return clusters;
}

void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    double scale = 0, cx = 0, cy = 0;
    double *colors = (double *)malloc(sizeof(double) * (k * 3));
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

    free(colors);
}

static void test()
{
    size_t size = 100000L;
    double startedAt = getWallTime();
    observation *observations =
        (observation *)malloc(sizeof(observation) * size);
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
    double generatedAt = getWallTime();
    int k = 5;
    cluster *clusters = kMeansCuda(observations, size, k);
    double clusteredAt = getWallTime();
    printEPS(observations, size, clusters, k);
    double finishedAt = getWallTime();
    printBenchmark(size, k, generatedAt, clusteredAt, finishedAt, startedAt);
    free(observations);
    free(clusters);
}

void test2()
{
    size_t size = 5000000L;
    double startedAt = getWallTime();
    observation *observations =
        (observation *)malloc(sizeof(observation) * size);
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
    double generatedAt = getWallTime();
    int k = 11;
    cluster *clusters = kMeansCuda(observations, size, k);
    double clusteredAt = getWallTime();
    printEPS(observations, size, clusters, k);
    double finishedAt = getWallTime();
    printBenchmark(size, k, generatedAt, clusteredAt, finishedAt, startedAt);
    free(observations);
    free(clusters);
}

int main()
{
#if RANDOM_SEED
    srand(RANDOM_SEED);
#else
    srand((unsigned int)time(NULL));
#endif
    // test();
    test2();
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
