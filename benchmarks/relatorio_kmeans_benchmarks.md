# Relatorio - K-Means Sequencial, OpenMP e CUDA

## Objetivo

Este relatorio compara tres versoes do algoritmo K-Means:

- `src/Kmeans.c`: versao sequencial em CPU.
- `src/Kmeans_cpu.c`: versao paralela em CPU usando OpenMP.
- `src/Kmeans_cuda.cu`: versao paralela em GPU usando CUDA.

O foco desta rodada e comparar as versoes com o mesmo tamanho de entrada,
compiladas sem otimizacao (`-O0`) e com a saida EPS descartada em `/dev/null`.

## Parametros

Todas as versoes executam `test2()`.

| Parametro | Valor |
| --- | ---: |
| Observacoes (`size`) | 5.000.000 |
| Clusters (`k`) | 11 |
| Raio maximo dos pontos | 20.00 |
| Saida | EPS no `stdout`, redirecionado para `/dev/null` |
| Otimizacao do compilador | `-O0` |
| Threads OpenMP | 32 |
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
gcc -O0 -fopenmp -DNUM_THREADS=32 src/Kmeans_cpu.c -o bin/Kmeans_cpu_omp -lm
time ./bin/Kmeans_cpu_omp > /dev/null
```

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
| OpenMP 32 threads | 9.590 | 82.268 | 0.602 |
| CUDA | 3.979 | 3.617 | 0.189 |

Interpretacao:

- `real`: tempo total percebido pelo usuario.
- `user`: soma de tempo de CPU em modo usuario. Em programas paralelos, pode ser
  maior que `real` porque soma o uso de varias threads.
- `sys`: tempo de CPU gasto pelo sistema operacional.

Na versao OpenMP, `user` ficou muito maior que `real` porque 32 threads executam
em paralelo e o `time` soma o tempo de CPU consumido por todas elas.

## Speedup pelo tempo real

O speedup foi calculado usando:

```text
Speedup = tempo_sequencial / tempo_paralelo
```

| Versao | real (s) | Speedup vs sequencial |
| --- | ---: | ---: |
| Sequencial | 19.531 | 1.00x |
| OpenMP 32 threads | 9.590 | 2.04x |
| CUDA | 3.979 | 4.91x |

Calculos:

```text
Speedup OpenMP = 19.531 / 9.590 = 2.04
Speedup CUDA = 19.531 / 3.979 = 4.91
CUDA vs OpenMP = 9.590 / 3.979 = 2.41
```

Nesta rodada, a versao CUDA foi aproximadamente 4.91x mais rapida que a
sequencial e 2.41x mais rapida que a OpenMP no tempo total percebido.

## Benchmark interno - OpenMP

Saida da versao OpenMP:

```text
Benchmark: size=5000000 k=11 threads=32
geracao_dados: 0.237796 s
kmeans: 4.815350 s
printEPS: 1.582542 s
total_medido: 6.635688 s
```

Percentual por etapa:

| Etapa | Tempo (s) | Percentual |
| --- | ---: | ---: |
| geracao_dados | 0.237796 | 3.58% |
| kmeans | 4.815350 | 72.57% |
| printEPS | 1.582542 | 23.85% |
| total_medido | 6.635688 | 100.00% |

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

## Comparacao por etapa

| Etapa | OpenMP 32 threads (s) | CUDA (s) | Relacao |
| --- | ---: | ---: | ---: |
| geracao_dados | 0.237796 | 0.287319 | OpenMP 1.21x mais rapida |
| kmeans | 4.815350 | 1.750357 | CUDA 2.75x mais rapida |
| printEPS | 1.582542 | 1.751729 | OpenMP 1.11x mais rapida |
| total_medido interno | 6.635688 | 3.789405 | CUDA 1.75x mais rapida |

A etapa principal do algoritmo (`kmeans`) foi 2.75x mais rapida na GPU do que
na versao OpenMP com 32 threads, nesta configuracao sem otimizacao.

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

## Conclusao

Com 5.000.000 observacoes, `k = 11`, compilacao `-O0` e saida descartada em
`/dev/null`, a versao CUDA apresentou o melhor tempo total:

| Versao | real (s) |
| --- | ---: |
| Sequencial | 19.531 |
| OpenMP 32 threads | 9.590 |
| CUDA | 3.979 |

A GPU reduziu significativamente o tempo da etapa principal do K-Means, levando
`kmeans_cuda` a 1.750357 s contra 4.815350 s da versao OpenMP. Ainda assim,
`printEPS()` continua relevante no tempo total, especialmente porque a saida EPS
e gerada para todos os 5.000.000 pontos.
