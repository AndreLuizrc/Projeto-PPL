/* ---- Tempo de execucao sequencial --------
    real    0m17.850s
    user    0m0.015s
    sys     0m0.061s
*/

/* ---- Tempo paralelo 1 thread --------
    real    0m17.833s
    user    0m0.015s
    sys     0m0.093s
*/

/* ---- Tempo paralelo 2 threads --------
    real    0m17.132s
    user    0m0.030s
    sys     0m0.046s
*/

/* ---- Tempo paralelo 4 threads --------
    real    0m16.896s
    ser    0m0.015s
    sys     0m0.077s
*/

/* ---- Tempo paralelo 8 threads --------
    real    0m17.000s
    user    0m0.030s
    sys     0m0.046s
*/

/* ---- Tempo paralelo 16 threads --------
    real    0m16.639s
    user    0m0.061s
    sys     0m0.046s
*/

/* ---- Tempo paralelo 32 threads --------
    real    0m16.409s
    user    0m0.030s
    sys     0m0.015s
*/

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time, clock */

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_set_num_threads(n) ((void)0)
#define omp_set_dynamic(n) ((void)0)
#endif

/* PARALLELIZACAO: tag para definir o numero de threads usadas pelo OpenMP.
   Pode ser alterada aqui ou sobrescrita na compilacao, por exemplo:
   gcc -fopenmp -DNUM_THREADS=8 Kmeans_cpu.c -o Kmeans_cpu */
#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

/* RANDOM_SEED=0 usa time(NULL). Para comparar execucoes, use uma semente fixa:
   gcc -fopenmp -DNUM_THREADS=8 -DRANDOM_SEED=1 Kmeans_cpu.c -o Kmeans_cpu */
#ifndef RANDOM_SEED
#define RANDOM_SEED 0
#endif

static int getThreadCount()
{
    return NUM_THREADS > 0 ? NUM_THREADS : 1;
}

static double getWallTime()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static size_t getPaddedClusterStride(int k)
{
    /* MODIFICACAO PARA PARALELISMO:
       cada thread guarda somas parciais de todos os clusters. Em vez de usar
       exatamente k posicoes por thread, o bloco e alinhado para ocupar linhas
       de cache separadas, reduzindo false sharing entre threads. */
    const size_t cacheLineBytes = 64;
    size_t stride = (size_t)k;
    size_t elemsPerLine = cacheLineBytes / sizeof(double);

    if (elemsPerLine > 0)
    {
        size_t remainder = stride % elemsPerLine;
        if (remainder != 0)
        {
            stride += elemsPerLine - remainder;
        }
    }
    return stride;
}

static void printBenchmark(size_t size, int k, double generatedAt,
                           double clusteredAt, double finishedAt,
                           double startedAt)
{
    fprintf(stderr, "\nBenchmark: size=%zu k=%d threads=%d\n", size, k,
            getThreadCount());
    fprintf(stderr, "geracao_dados: %.6f s\n", generatedAt - startedAt);
    fprintf(stderr, "kmeans: %.6f s\n", clusteredAt - generatedAt);
    fprintf(stderr, "printEPS: %.6f s\n", finishedAt - clusteredAt);
    fprintf(stderr, "total_medido: %.6f s\n", finishedAt - startedAt);
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

int calculateNearst(observation *o, cluster clusters[], int k)
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
                       cluster *centroid)
{
    double sumX = 0;
    double sumY = 0;
    int numThreads = getThreadCount();

    centroid->count = size;

    /* MODIFICACAO DO ALGORITMO ORIGINAL:
       a soma do centroide, antes feita em um loop sequencial, foi transformada
       em uma reducao paralela. O OpenMP cria somas locais de sumX/sumY para
       cada thread e combina tudo no final.

       O #pragma divide o vetor de observacoes entre as threads; como cada
       iteracao altera apenas observations[i].group, nao ha disputa entre elas. */
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

cluster *kMeans(observation observations[], size_t size, int k)
{
    cluster *clusters = NULL;
    if (k <= 1)
    {
        /*
        If we have to cluster them only in one group
        then calculate centroid of observations and
        that will be a ingle cluster
        */
        clusters = (cluster *)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if ((size_t)k < size)
    {
        int numThreads = getThreadCount();

        /* MODIFICACAO DO ALGORITMO ORIGINAL:
           no recalculo dos centroides, varias observacoes podem pertencer ao
           mesmo cluster. Se todas as threads somassem diretamente em
           clusters[i], seria necessario usar locks/atomic, criando muita
           contencao.

           Para evitar isso, cada thread acumula seus proprios valores em
           partialX/partialY/partialCount. Depois essas somas parciais sao
           combinadas para formar os centroides finais. */
        size_t partialStride = getPaddedClusterStride(k);
        size_t partialLen = (size_t)numThreads * partialStride;
        double *partialX = (double *)calloc(partialLen, sizeof(double));
        double *partialY = (double *)calloc(partialLen, sizeof(double));
        size_t *partialCount = (size_t *)calloc(partialLen, sizeof(size_t));

        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        /* A atribuicao inicial dos grupos foi mantida sequencial porque rand()
           usa estado global; essa etapa acontece apenas uma vez e nao e o maior
           custo do algoritmo. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000; // Do until 99.99 percent points are in correct cluster
        do
        {
            changed = 0; // this variable stores change in clustering

            /* MODIFICACAO DO ALGORITMO ORIGINAL:
               cada iteracao do K-Means tinha varios loops sequenciais. Aqui uma
               unica regiao paralela executa os passos de limpeza, acumulacao,
               combinacao e reclassificacao, reduzindo o custo de criar/finalizar
               threads varias vezes.

               O reduction(+ : changed) permite que cada thread conte suas
               mudancas localmente; o OpenMP soma esses valores no fim da regiao. */
#pragma omp parallel num_threads(numThreads) reduction(+ : changed)
            {
                int threadId = omp_get_thread_num();
                size_t base = (size_t)threadId * partialStride;

                /* O #pragma omp for divide entre as threads a limpeza dos
                   acumuladores parciais usados nesta iteracao. */
#pragma omp for schedule(static)
                for (size_t idx = 0; idx < partialLen; idx++)
                {
                    partialX[idx] = 0;
                    partialY[idx] = 0;
                    partialCount[idx] = 0;
                }

                /* STEP 2*/
                /* MODIFICACAO DO ALGORITMO ORIGINAL:
                   em vez de atualizar o centroide global do cluster diretamente,
                   cada thread grava apenas no seu bloco parcial. Assim, duas
                   threads nunca escrevem na mesma posicao de
                   partialX/partialY/partialCount.

                   O #pragma omp for distribui as observacoes entre as threads. */
#pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;
                    partialX[base + group] += observations[j].x;
                    partialY[base + group] += observations[j].y;
                    partialCount[base + group]++;
                }

                /* MODIFICACAO DO ALGORITMO ORIGINAL:
                   foi adicionada uma etapa de reducao manual dos acumuladores
                   parciais. Cada iteracao deste loop calcula um cluster final,
                   somando os blocos de todas as threads.

                   O #pragma omp for paraleliza por cluster; como cada iteracao
                   escreve em um indice diferente de clusters, nao ha corrida. */
#pragma omp for schedule(static)
                for (int i = 0; i < k; i++)
                {
                    double sumX = 0;
                    double sumY = 0;
                    size_t count = 0;

                    for (int partialThread = 0; partialThread < numThreads;
                         partialThread++)
                    {
                        size_t idx =
                            (size_t)partialThread * partialStride + (size_t)i;
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
                /* MODIFICACAO DO ALGORITMO ORIGINAL:
                   a busca pelo centroide mais proximo foi paralelizada porque a
                   nova classe de uma observacao independe das demais. Cada
                   thread atualiza apenas observations[j].

                   O #pragma omp for divide as observacoes; a variavel changed
                   ja esta protegida pelo reduction da regiao paralela. */
#pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int nearest = calculateNearst(observations + j, clusters, k);
                    if (nearest != observations[j].group)
                    {
                        changed++;
                        observations[j].group = nearest;
                    }
                }
            }
        } while (changed > minAcceptedError); // Keep on grouping until we have
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
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* MODIFICACAO DO ALGORITMO ORIGINAL:
           no caso trivial, cada observacao vira seu proprio cluster. O loop foi
           paralelizado porque cada iteracao escreve em indices independentes de
           clusters e observations. */
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

    // free accquired memory
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
    int k = 5; // No of clusters
    cluster *clusters = kMeans(observations, size, k);
    double clusteredAt = getWallTime();
    printEPS(observations, size, clusters, k);
    double finishedAt = getWallTime();
    printBenchmark(size, k, generatedAt, clusteredAt, finishedAt, startedAt);
    // Free the accquired memory
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
    int k = 11; // No of clusters
    cluster *clusters = kMeans(observations, size, k);
    double clusteredAt = getWallTime();
    printEPS(observations, size, clusters, k);
    double finishedAt = getWallTime();
    printBenchmark(size, k, generatedAt, clusteredAt, finishedAt, startedAt);
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
    omp_set_dynamic(0);
    omp_set_num_threads(getThreadCount());
#if RANDOM_SEED
    srand(RANDOM_SEED);
#else
    srand(time(NULL));
#endif
    // test();
    test2();
    return 0;
}
