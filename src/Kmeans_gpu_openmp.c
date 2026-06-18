/* ---- Versao paralela em GPU com OpenMP offload --------
   Para executar de fato na GPU, compile com suporte a OpenMP target/offload no
   compilador usado. Em compiladores sem dispositivo configurado, o OpenMP pode
   executar as regioes target no host como fallback.

   Exemplo GCC/NVIDIA, dependendo da instalacao:
   gcc -O2 -fopenmp -foffload=nvptx-none Kmeans_gpu_openmp.c -o Kmeans_gpu_openmp.exe -lm
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
#define omp_get_default_device() 0
#define omp_get_num_devices() 0
#define omp_get_wtime() ((double)clock() / CLOCKS_PER_SEC)
#define omp_set_default_device(n) ((void)0)
#endif

/* PARALLELIZACAO GPU:
   NUM_TEAMS controla quantos grupos de threads OpenMP sao solicitados para o
   dispositivo. Pode ser sobrescrito na compilacao:
   gcc -fopenmp -DNUM_TEAMS=512 Kmeans_gpu_openmp.c -o Kmeans_gpu_openmp */
#ifndef NUM_TEAMS
#define NUM_TEAMS 256
#endif

/* PARALLELIZACAO GPU:
   THREAD_LIMIT controla o limite de threads por team no dispositivo. O valor
   ideal depende da GPU e do compilador OpenMP usado. */
#ifndef THREAD_LIMIT
#define THREAD_LIMIT 128
#endif

/* Limite usado para manter acumuladores locais por thread no device. O teste
   principal usa k = 11; se k passar deste limite, o codigo usa o caminho
   generico com acumuladores parciais em memoria global. */
#ifndef LOCAL_CLUSTER_LIMIT
#define LOCAL_CLUSTER_LIMIT 16
#endif

/* RANDOM_SEED=0 usa time(NULL). Para comparar execucoes, use uma semente fixa:
   gcc -fopenmp -DRANDOM_SEED=1 Kmeans_gpu_openmp.c -o Kmeans_gpu_openmp */
#ifndef RANDOM_SEED
#define RANDOM_SEED 0
#endif

static double getWallTime()
{
    return omp_get_wtime();
}

static int getDeviceCount()
{
    return omp_get_num_devices();
}

static int getDefaultDevice()
{
    return omp_get_default_device();
}

static void printBenchmark(size_t size, int k, double generatedAt,
                           double clusteredAt, double finishedAt,
                           double startedAt)
{
    fprintf(stderr,
            "\nBenchmark: size=%zu k=%d devices=%d default_device=%d teams=%d "
            "thread_limit=%d\n",
            size, k, getDeviceCount(), getDefaultDevice(), NUM_TEAMS,
            THREAD_LIMIT);
    fprintf(stderr, "geracao_dados: %.6f s\n", generatedAt - startedAt);
    fprintf(stderr, "kmeans_gpu_openmp: %.6f s\n", clusteredAt - generatedAt);
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

    centroid->count = size;

    /* MODIFICACAO PARA GPU:
       no caso k <= 1, o custo nao e dominante para o benchmark principal.
       Mantemos esta etapa no host para evitar gerar reducoes compostas no
       target NVPTX, que podem exigir atomics de 16 bytes em algumas versoes do
       GCC offload. */
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
        /* MODIFICACAO PARA GPU:
           os centroides foram separados em tres vetores simples. Esse formato
           e mais facil de mapear para o dispositivo do que atualizar um vetor
           de structs diretamente. */
        double *clusterX = (double *)calloc((size_t)k, sizeof(double));
        double *clusterY = (double *)calloc((size_t)k, sizeof(double));
        size_t *clusterCount = (size_t *)calloc((size_t)k, sizeof(size_t));
        int partialBins = NUM_TEAMS * THREAD_LIMIT;
        size_t partialLen = (size_t)partialBins * (size_t)k;
        double *partialX = (double *)calloc(partialLen, sizeof(double));
        double *partialY = (double *)calloc(partialLen, sizeof(double));
        unsigned int *partialCount =
            (unsigned int *)calloc(partialLen, sizeof(unsigned int));

        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        /* A atribuicao inicial dos grupos foi mantida no host porque rand()
           usa estado global e nao e portavel dentro de uma regiao target.
           Essa etapa acontece apenas uma vez; o custo dominante fica nas
           iteracoes seguintes, que foram movidas para a GPU. */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        unsigned int changed = 0;
        unsigned int minAcceptedError =
            (unsigned int)(size /
                           10000); // Do until 99.99 percent points are in correct cluster

        /* MODIFICACAO PARA GPU:
           mantemos observations e os vetores dos centroides alocados no
           dispositivo durante todo o loop do K-Means. Assim evitamos copiar
           1.000.000 de observacoes entre CPU e GPU a cada iteracao. */
#pragma omp target data map(tofrom : observations[0 : size]) map(alloc : clusterX[0 : k], clusterY[0 : k], clusterCount[0 : k], partialX[0 : partialLen], partialY[0 : partialLen], partialCount[0 : partialLen])
        {
            do
            {
                changed = 0; // this variable stores change in clustering

                /* STEP 2*/
                /* MODIFICACAO PARA GPU:
                   cada thread do dispositivo acumula em uma faixa privada de
                   partialX/partialY/partialCount. Isso evita atomics no NVPTX
                   e tambem evita varrer todas as observacoes varias vezes por
                   cluster. */
#pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT)
                for (size_t idx = 0; idx < partialLen; idx++)
                {
                    partialX[idx] = 0;
                    partialY[idx] = 0;
                    partialCount[idx] = 0;
                }

                if (k <= LOCAL_CLUSTER_LIMIT)
                {
                    /* Caminho rapido para o caso do projeto: cada thread
                       acumula todos os clusters em registradores/memoria local
                       e grava seus parciais uma unica vez no fim do bloco. */
#pragma omp target teams distribute num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT)
                    for (int chunk = 0; chunk < NUM_TEAMS; chunk++)
                    {
                        size_t chunkSize =
                            (size + (size_t)NUM_TEAMS - 1) / (size_t)NUM_TEAMS;
                        size_t begin = (size_t)chunk * chunkSize;
                        size_t end = begin + chunkSize;

                        if (end > size)
                        {
                            end = size;
                        }

#pragma omp parallel num_threads(THREAD_LIMIT)
                        {
                            int bin =
                                chunk * THREAD_LIMIT + omp_get_thread_num();
                            double localX[LOCAL_CLUSTER_LIMIT];
                            double localY[LOCAL_CLUSTER_LIMIT];
                            unsigned int localCount[LOCAL_CLUSTER_LIMIT];

                            for (int i = 0; i < k; i++)
                            {
                                localX[i] = 0;
                                localY[i] = 0;
                                localCount[i] = 0;
                            }

#pragma omp for
                            for (size_t j = begin; j < end; j++)
                            {
                                int group = observations[j].group;
                                localX[group] += observations[j].x;
                                localY[group] += observations[j].y;
                                localCount[group]++;
                            }

                            if (bin < partialBins)
                            {
                                for (int i = 0; i < k; i++)
                                {
                                    size_t idx =
                                        (size_t)bin * (size_t)k + (size_t)i;
                                    partialX[idx] = localX[i];
                                    partialY[idx] = localY[i];
                                    partialCount[idx] = localCount[i];
                                }
                            }
                        }
                    }
                }
                else
                {
                    /* Caminho generico para k maior que LOCAL_CLUSTER_LIMIT:
                       ainda evita atomics, mas escreve os parciais em memoria
                       global durante a varredura. */
#pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT)
                    for (size_t j = 0; j < size; j++)
                    {
                        int group = observations[j].group;
                        int bin = omp_get_team_num() * THREAD_LIMIT +
                                  omp_get_thread_num();

                        if (bin < partialBins)
                        {
                            size_t idx =
                                (size_t)bin * (size_t)k + (size_t)group;
                            partialX[idx] += observations[j].x;
                            partialY[idx] += observations[j].y;
                            partialCount[idx]++;
                        }
                    }
                }

                /* MODIFICACAO PARA GPU:
                   combina os acumuladores privados em um centroide por
                   cluster. O custo e proporcional a NUM_TEAMS * THREAD_LIMIT *
                   k, bem menor que repetir a varredura dos pontos por cluster. */
#pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT)
                for (int i = 0; i < k; i++)
                {
                    double sumX = 0;
                    double sumY = 0;
                    unsigned int count = 0;

                    for (int bin = 0; bin < partialBins; bin++)
                    {
                        size_t idx = (size_t)bin * (size_t)k + (size_t)i;
                        sumX += partialX[idx];
                        sumY += partialY[idx];
                        count += partialCount[idx];
                    }

                    clusterCount[i] = count;
                    if (count > 0)
                    {
                        clusterX[i] = sumX / count;
                        clusterY[i] = sumY / count;
                    }
                    else
                    {
                        clusterX[i] = 0;
                        clusterY[i] = 0;
                    }
                }

                /* STEP 3 and 4 */
                /* MODIFICACAO PARA GPU:
                   a busca do centroide mais proximo foi incorporada ao proprio
                   loop target para evitar chamada de funcao dentro do
                   dispositivo. Cada thread calcula uma observacao de forma
                   independente e a reducao soma quantas mudaram de grupo. */
#pragma omp target teams distribute parallel for reduction(+ : changed) num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT)
                for (size_t j = 0; j < size; j++)
                {
                    double minD = DBL_MAX;
                    int nearest = -1;

                    for (int i = 0; i < k; i++)
                    {
                        double dx = clusterX[i] - observations[j].x;
                        double dy = clusterY[i] - observations[j].y;
                        double dist = dx * dx + dy * dy;

                        if (dist < minD)
                        {
                            minD = dist;
                            nearest = i;
                        }
                    }

                    if (nearest != observations[j].group)
                    {
                        changed++;
                        observations[j].group = nearest;
                    }
                }
            } while (changed > minAcceptedError); // Keep on grouping until we
                                                  // have got almost best
                                                  // clustering

            /* MODIFICACAO PARA GPU:
               os centroides finais sao copiados de volta uma unica vez, apos a
               convergencia. As observacoes voltam automaticamente ao sair da
               regiao target data por causa do map(tofrom). */
#pragma omp target update from(clusterX[0 : k], clusterY[0 : k], clusterCount[0 : k])
        }

        for (int i = 0; i < k; i++)
        {
            clusters[i].x = clusterX[i];
            clusters[i].y = clusterY[i];
            clusters[i].count = clusterCount[i];
        }

        free(clusterX);
        free(clusterY);
        free(clusterCount);
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
        /* MODIFICACAO PARA GPU:
           no caso trivial, cada observacao vira seu proprio cluster. O loop foi
           movido para a GPU porque cada iteracao escreve em indices
           independentes de clusters e observations. */
#pragma omp target teams distribute parallel for map(tofrom : observations[0 : size], clusters[0 : k]) num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT)
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
    /* PARALLELIZACAO GPU:
       quando existir dispositivo OpenMP disponivel, seleciona o device padrao
       0. Se nao houver GPU configurada, a execucao target fica a cargo do
       fallback do runtime/compilador. */
    if (getDeviceCount() > 0)
    {
        omp_set_default_device(0);
    }

#if RANDOM_SEED
    srand(RANDOM_SEED);
#else
    srand(time(NULL));
#endif
    // test();
    test2();
    return 0;
}