# Relatorio - K-Means Sequencial, OpenMP, OpenMP GPU e CUDA

## Objetivo

Este relatorio compara quatro versoes do algoritmo K-Means:

- `src/Kmeans.c`: versao sequencial em CPU.
- `src/Kmeans_cpu.c`: versao paralela em CPU usando OpenMP.
- `src/Kmeans_gpu_openmp.c`: versao paralela em GPU usando OpenMP target/offload.
- `src/Kmeans_cuda.cu`: versao paralela em GPU usando CUDA.

O foco desta rodada e comparar as versoes com o mesmo tamanho de entrada,
compiladas sem otimizacao (`-O0`) e com a saida EPS descartada em `/dev/null`.
Tambem foi adicionada uma medicao da versao OpenMP GPU compilada com `-O2`,
porque o offload NVPTX do GCC exigiu flags especificas de compilacao.

## Parametros

Todas as versoes executam `test2()`.

| Parametro | Valor |
| --- | ---: |
| Observacoes (`size`) | 5.000.000 |
| Clusters (`k`) | 11 |
| Raio maximo dos pontos | 20.00 |
| Saida | EPS no `stdout`, redirecionado para `/dev/null` |
| Otimizacao do compilador | `-O0` na rodada principal; OpenMP GPU com `-O2` |
| Threads OpenMP testadas | 1, 2, 4, 8, 16 e 32 |
| OpenMP GPU | `target/offload` com `devices=1` |
| OpenMP GPU teams/thread_limit | 256 teams, 128 threads por team |
| Bloco CUDA | 256 threads |
| Arquitetura CUDA | `sm_89` |

`size` representa a quantidade de pontos 2D gerados e agrupados pelo K-Means.
Cada ponto possui coordenadas `x`, `y` e um campo `group` indicando o cluster.

## Comandos usados

Versao sequencial:

```bash
gcc -O0 src/Kmeans.c -o bin/Kmeans_seq_O0 -lm
time ./bin/Kmeans_seq_O0 > /dev/null
```

Versao OpenMP:

```bash
gcc -O0 -fopenmp -DNUM_THREADS=<threads> src/Kmeans_cpu.c -o bin/Kmeans_cpu_omp -lm
time ./bin/Kmeans_cpu_omp > /dev/null
```

Nos testes OpenMP, `<threads>` foi variado entre 1, 2, 4, 8, 16 e 32.

Versao OpenMP GPU:

```bash
gcc -O2 -fopenmp \
  -fno-stack-protector \
  -fcf-protection=none \
  -foffload=nvptx-none \
  -foffload-options=nvptx-none="-fno-stack-protector -fcf-protection=none" \
  src/Kmeans_gpu_openmp.c -o bin/Kmeans_gpu_openmp -lm

./bin/Kmeans_gpu_openmp > /dev/null
```

As flags `-fno-stack-protector` e `-fcf-protection=none` foram necessarias
porque o backend NVPTX do GCC nao aceita essas protecoes quando elas aparecem no
codigo compilado para o dispositivo.

Versao CUDA:

```bash
nvcc -O0 -arch=sm_89 src/Kmeans_cuda.cu -o bin/Kmeans_cuda
time ./bin/Kmeans_cuda > /dev/null
```

O aviso do CUDA sobre a funcao `test` nao ser usada nao afeta a execucao, pois
o `main` chama `test2()`.

## Tempo externo (`time`)

| Versao | real (s) | user (s) | sys (s) |
| --- | ---: | ---: | ---: |
| Sequencial | 19.531 | 21.208 | 0.083 |
| OpenMP 1 thread | 24.140 | 23.053 | 0.066 |
| OpenMP 2 threads | 11.881 | 23.798 | 0.057 |
| OpenMP 4 threads | 5.828 | 20.024 | 0.044 |
| OpenMP 8 threads | 8.608 | 61.897 | 0.071 |
| OpenMP 16 threads | 7.470 | 102.655 | 0.106 |
| OpenMP 32 threads | 9.590 | 82.268 | 0.602 |
| CUDA | 3.979 | 3.617 | 0.189 |

Interpretacao:

- `real`: tempo total percebido pelo usuario.
- `user`: soma de tempo de CPU em modo usuario. Em programas paralelos, pode ser
  maior que `real` porque soma o uso de varias threads.
- `sys`: tempo de CPU gasto pelo sistema operacional.

Nas versoes OpenMP com varias threads, `user` ficou maior que `real` porque o
`time` soma o tempo de CPU consumido por todas as threads.

## Speedup pelo tempo real

O speedup foi calculado usando:

```text
Speedup = tempo_sequencial / tempo_paralelo
```

| Versao | real (s) | Speedup vs sequencial |
| --- | ---: | ---: |
| Sequencial | 19.531 | 1.00x |
| OpenMP 1 thread | 24.140 | 0.81x |
| OpenMP 2 threads | 11.881 | 1.64x |
| OpenMP 4 threads | 5.828 | 3.35x |
| OpenMP 8 threads | 8.608 | 2.27x |
| OpenMP 16 threads | 7.470 | 2.61x |
| OpenMP 32 threads | 9.590 | 2.04x |
| CUDA | 3.979 | 4.91x |

Calculos:

```text
Speedup OpenMP 4 threads = 19.531 / 5.828 = 3.35
Speedup OpenMP 32 threads = 19.531 / 9.590 = 2.04
Speedup CUDA = 19.531 / 3.979 = 4.91
CUDA vs melhor OpenMP = 5.828 / 3.979 = 1.46
```

Nesta rodada, a melhor configuracao OpenMP pelo tempo real foi com 4 threads.
A versao CUDA foi aproximadamente 4.91x mais rapida que a sequencial e 1.46x
mais rapida que a melhor execucao OpenMP no tempo total percebido.

## Benchmark interno - OpenMP

Saidas internas da versao OpenMP:

| Threads | geracao_dados (s) | kmeans (s) | printEPS (s) | total_medido (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.258212 | 19.260821 | 1.671872 | 21.190905 |
| 2 | 0.262423 | 10.045937 | 1.558816 | 11.867175 |
| 4 | 0.239392 | 4.234563 | 1.341754 | 5.815710 |
| 8 | 0.240066 | 7.059653 | 1.294758 | 8.594477 |
| 16 | 0.263665 | 5.896827 | 1.297442 | 7.457934 |
| 32 | 0.237796 | 4.815350 | 1.582542 | 6.635688 |

Speedup da etapa `kmeans` em relacao ao OpenMP com 1 thread:

| Threads | kmeans (s) | Speedup do kmeans |
| ---: | ---: | ---: |
| 1 | 19.260821 | 1.00x |
| 2 | 10.045937 | 1.92x |
| 4 | 4.234563 | 4.55x |
| 8 | 7.059653 | 2.73x |
| 16 | 5.896827 | 3.27x |
| 32 | 4.815350 | 4.00x |

A melhor configuracao OpenMP nessa rodada foi 4 threads, tanto no tempo real
quanto na etapa `kmeans`. O resultado nao cresceu de forma monotonicamente
melhor com mais threads, o que indica overhead, contencao, custo de sincronizacao
ou disputa por recursos da CPU.

## Benchmark interno - CUDA

Saida da versao CUDA:

```text
Benchmark CUDA: size=5000000 k=11 block=256
geracao_dados: 0.287319 s
kmeans_cuda: 1.750357 s
printEPS: 1.751729 s
total_medido: 3.789405 s
```

Percentual por etapa:

| Etapa | Tempo (s) | Percentual |
| --- | ---: | ---: |
| geracao_dados | 0.287319 | 7.58% |
| kmeans_cuda | 1.750357 | 46.19% |
| printEPS | 1.751729 | 46.23% |
| total_medido | 3.789405 | 100.00% |

## Benchmark interno - OpenMP GPU

Saida da versao OpenMP target/offload em GPU:

```text
Benchmark: size=5000000 k=11 devices=1 default_device=0 teams=256 thread_limit=128
geracao_dados: 0.207701 s
kmeans_gpu_openmp: 12.527001 s
printEPS: 1.412957 s
total_medido: 14.147659 s
```

Percentual por etapa:

| Etapa | Tempo (s) | Percentual |
| --- | ---: | ---: |
| geracao_dados | 0.207701 | 1.47% |
| kmeans_gpu_openmp | 12.527001 | 88.54% |
| printEPS | 1.412957 | 9.99% |
| total_medido | 14.147659 | 100.00% |

O campo `devices=1` indica que o runtime OpenMP encontrou um dispositivo para
offload. Mesmo assim, nesta rodada a versao OpenMP GPU ficou mais lenta que a
versao CUDA e que a melhor versao OpenMP em CPU. Isso sugere overhead do runtime
OpenMP offload, custo das regioes `target` e/ou menor eficiencia do padrao de
acumulacao em comparacao com os kernels CUDA especializados.

## Comparacao por etapa

| Etapa | OpenMP 32 threads (s) | CUDA (s) | Relacao |
| --- | ---: | ---: | ---: |
| geracao_dados | 0.237796 | 0.287319 | OpenMP 1.21x mais rapida |
| kmeans | 4.815350 | 1.750357 | CUDA 2.75x mais rapida |
| printEPS | 1.582542 | 1.751729 | OpenMP 1.11x mais rapida |
| total_medido interno | 6.635688 | 3.789405 | CUDA 1.75x mais rapida |

Comparando com a melhor configuracao OpenMP por tempo real:

| Etapa | OpenMP 4 threads (s) | CUDA (s) | Relacao |
| --- | ---: | ---: | ---: |
| geracao_dados | 0.239392 | 0.287319 | OpenMP 1.20x mais rapida |
| kmeans | 4.234563 | 1.750357 | CUDA 2.42x mais rapida |
| printEPS | 1.341754 | 1.751729 | OpenMP 1.31x mais rapida |
| total_medido interno | 5.815710 | 3.789405 | CUDA 1.53x mais rapida |

A etapa principal do algoritmo (`kmeans`) foi 2.42x mais rapida na GPU do que
na melhor configuracao OpenMP medida, nesta configuracao sem otimizacao.

Comparando CUDA com OpenMP GPU:

| Etapa | OpenMP GPU (s) | CUDA (s) | Relacao |
| --- | ---: | ---: | ---: |
| geracao_dados | 0.207701 | 0.287319 | OpenMP GPU 1.38x mais rapida |
| kmeans | 12.527001 | 1.750357 | CUDA 7.16x mais rapida |
| printEPS | 1.412957 | 1.751729 | OpenMP GPU 1.24x mais rapida |
| total_medido interno | 14.147659 | 3.789405 | CUDA 3.73x mais rapida |

Essa comparacao deve ser lida com cuidado porque a versao OpenMP GPU foi
compilada com `-O2`, enquanto a rodada principal documentada para sequencial,
OpenMP CPU e CUDA usa `-O0`.

## Sobre `/dev/null`

O redirecionamento para `/dev/null` evita gravar um arquivo EPS gigante no disco,
mas nao remove o custo de `printEPS()`. A funcao ainda percorre os pontos,
formata os comandos PostScript e escreve no `stdout`; o sistema operacional
apenas descarta essa saida.

Por isso, os tempos ainda incluem parte do custo de geracao da visualizacao. Para
medir apenas o algoritmo K-Means, seria necessario criar uma opcao no codigo para
nao chamar `printEPS()`.

## O que a saida EPS representa

O EPS gerado pelo programa e uma imagem vetorial dos clusters:

- `R G B setrgbcolor`: escolhe a cor de um cluster.
- `x y c`: desenha uma observacao naquela cor.
- `0 setgray x y s`: desenha o centroide do cluster.
- `%%EOF`: finaliza o arquivo.

As coordenadas impressas nao sao as coordenadas originais dos pontos. A funcao
`printEPS()` calcula os limites dos dados e escala tudo para caber em uma imagem
de 400x400.

## Etapa final - Conclusoes

Com 5.000.000 observacoes, `k = 11`, compilacao `-O0` e saida descartada em
`/dev/null`, a versao CUDA apresentou o melhor tempo total entre as versoes da
rodada principal. A versao OpenMP GPU foi medida separadamente com `-O2`, pois o
offload NVPTX exigiu flags especificas de compilacao.

| Versao | tempo total (s) | Origem |
| --- | ---: | --- |
| Sequencial | 19.531 | `time real` |
| OpenMP 4 threads | 5.828 | `time real` |
| OpenMP GPU target | 14.148 | benchmark interno |
| CUDA | 3.979 | `time real` |

Principais conclusoes:

- A versao sequencial serviu como base de comparacao, com `real = 19.531 s`.
- A melhor versao OpenMP em CPU foi com 4 threads, chegando a `real = 5.828 s`,
  ou seja, speedup de 3.35x sobre a sequencial.
- A versao CUDA teve o melhor desempenho geral, com `real = 3.979 s` e speedup
  de 4.91x sobre a sequencial.
- Na etapa principal do algoritmo, `kmeans_cuda` levou 1.750357 s, enquanto a
  melhor versao OpenMP CPU levou 4.234563 s. Assim, CUDA foi 2.42x mais rapida
  na parte central do K-Means.
- A versao OpenMP GPU encontrou um dispositivo (`devices=1`), confirmando que o
  runtime reconheceu uma GPU para offload. Mesmo assim, `kmeans_gpu_openmp`
  levou 12.527001 s, ficando mais lenta que CUDA e que a melhor versao OpenMP
  em CPU.
- O resultado da OpenMP GPU mostra que usar GPU nao garante ganho
  automaticamente. O desempenho depende do custo de offload, da forma de
  acumulacao dos centroides, das sincronizacoes e da eficiencia do compilador
  para gerar codigo para o dispositivo.
- A funcao `printEPS()` continua tendo impacto relevante no tempo total. Mesmo
  com a saida redirecionada para `/dev/null`, o programa ainda percorre os
  pontos e formata todos os comandos PostScript.

Portanto, para este conjunto de testes, a implementacao CUDA foi a alternativa
mais eficiente. A versao OpenMP em CPU apresentou bom ganho com poucas threads,
mas perdeu eficiencia ao aumentar demais a quantidade de threads. A versao
OpenMP GPU funcionou corretamente com offload, porem ainda precisa de
otimizacoes para competir com CUDA.

Como melhoria futura para uma comparacao mais precisa, recomenda-se adicionar
uma opcao para desativar `printEPS()` durante os benchmarks. Assim seria possivel
medir apenas o tempo do algoritmo K-Means, sem o custo de geracao da
visualizacao EPS.
