# Relatorio - K-Means

## Integrantes

- Nome 1:
- Nome 2:
- Nome 3:

## Objetivo

Descrever e comparar o desempenho das diferentes versoes do algoritmo
K-Means implementadas no projeto, considerando tempo de execucao e speedup.

## Codigos avaliados

- `Kmeans.c`: versao sequencial.
- `Kmeans_cpu.c`: versao paralela em CPU com OpenMP.

Observacao: atualmente o `main` executa a funcao `test2()`, com 1.000.000
observacoes e `k = 11`. A saida do programa e gerada em formato EPS no
stdout e, nos testes realizados, foi exibida no proprio terminal.

## Como compilar

No terminal, entre na pasta do projeto:

```bash
cd Projeto-PPL
```

Compilacao da versao sequencial:

```bash
gcc Kmeans.c -o Kmeans.exe -lm
```

Compilacao da versao paralela com OpenMP:

```bash
gcc -fopenmp Kmeans_cpu.c -o Kmeans_cpu_omp.exe -lm
```

Compilacao da versao paralela definindo a quantidade de threads:

```bash
gcc -O2 -fopenmp -DNUM_THREADS=8 Kmeans_cpu.c -o Kmeans_cpu_omp.exe -lm
```

Substitua `8` pela quantidade de threads desejada.

## Como rodar

Execucao da versao sequencial:

```bash
./Kmeans.exe
```

Execucao da versao paralela:

```bash
./Kmeans_cpu_omp.exe
```

No PowerShell, para medir o tempo de execucao:

```powershell
Measure-Command { .\Kmeans.exe }
Measure-Command { .\Kmeans_cpu_omp.exe }
```

No Linux ou Git Bash, para medir o tempo de execucao:

```bash
time ./Kmeans.exe
time ./Kmeans_cpu_omp.exe
```

## Ambiente de testes

- Processador:
- Quantidade de nucleos/threads:
- Memoria RAM:
- Sistema operacional:
- Compilador e versao:
- Flags de compilacao:

## Metodologia

Os testes foram realizados da seguinte forma:

- Cada versao foi executada 3 vezes.
- Os valores apresentados na tabela correspondem a media das 3 execucoes.
- A saida EPS foi mantida no proprio terminal, sem redirecionamento para arquivo.
- O tempo considerado foi o tempo real de execucao, incluindo a impressao da saida.
- Tambem foi feito um benchmark interno separando o tempo de geracao dos dados,
  execucao do K-Means e impressao da saida EPS.
- O mesmo tamanho de entrada foi usado em todas as versoes.

## Tempos de execucao

Os tempos abaixo correspondem a media de 3 execucoes para cada configuracao.

| Versao | Threads | Tempo medio (s) |
| --- | ---: | ---: |
| Sequencial (`Kmeans.c`) | 1 | 17.850 |
| Paralela (`Kmeans_cpu.c`) | 1 | 17.833 |
| Paralela (`Kmeans_cpu.c`) | 2 | 17.132 |
| Paralela (`Kmeans_cpu.c`) | 4 | 16.896 |
| Paralela (`Kmeans_cpu.c`) | 8 | 17.000 |
| Paralela (`Kmeans_cpu.c`) | 16 | 16.639 |
| Paralela (`Kmeans_cpu.c`) | 32 | 16.409 |

## Calculo de speedup

O speedup mede quantas vezes a versao paralela foi mais rapida que a versao
sequencial:

```text
Speedup = tempo_sequencial / tempo_paralelo
```

Use a media dos tempos da tabela anterior para calcular:

| Versao | Threads | Speedup |
| --- | ---: | ---: |
| Sequencial | 1 | 1.00 |
| Paralela | 1 | 1.00 |
| Paralela | 2 | 1.04 |
| Paralela | 4 | 1.06 |
| Paralela | 8 | 1.05 |
| Paralela | 16 | 1.07 |
| Paralela | 32 | 1.09 |

Exemplo usando o tempo sequencial registrado:

```text
Speedup_8_threads = 17.850 / 17.000 = 1.05
```

## Benchmark por etapa

Como a funcao `printEPS()` imprime a saida em formato EPS de forma sequencial,
ela pode dominar o tempo total de execucao e esconder o ganho obtido na parte
paralelizada do algoritmo. Por isso, tambem foi medido separadamente o tempo de
cada etapa principal do programa.

| Threads | geracao_dados (s) | kmeans (s) | printEPS (s) | total_medido (s) | printEPS no total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.031 | 1.088 | 16.241 | 17.360 | 93.55% |
| 2 | 0.031 | 0.796 | 16.381 | 17.208 | 95.19% |
| 4 | 0.028 | 0.753 | 16.112 | 16.893 | 95.38% |
| 8 | 0.029 | 0.381 | 16.873 | 17.283 | 97.63% |
| 16 | 0.028 | 0.184 | 16.688 | 16.900 | 98.75% |
| 32 | 0.030 | 0.137 | 16.754 | 16.921 | 99.01% |

Considerando apenas a etapa `kmeans`, o speedup fica:

| Threads | Tempo kmeans (s) | Speedup do kmeans |
| ---: | ---: | ---: |
| 1 | 1.088 | 1.00 |
| 2 | 0.796 | 1.37 |
| 4 | 0.753 | 1.45 |
| 8 | 0.381 | 2.86 |
| 16 | 0.184 | 5.91 |
| 32 | 0.137 | 7.94 |

## Analise dos resultados

A tabela de tempo real mostra um ganho pequeno quando a versao paralela e
comparada com a versao sequencial. O melhor tempo total registrado foi com 32
threads, chegando a 16.409 s contra 17.850 s da versao sequencial, o que
representa speedup de 1.09.

O benchmark por etapa explica esse resultado: a funcao `printEPS()` consome a
maior parte do tempo total, ficando entre 93.55% e 99.01% da execucao. Como essa
parte e sequencial e faz muita impressao no terminal, ela limita o ganho visivel
no tempo total.

Quando se observa apenas a etapa `kmeans`, que e a parte paralelizada, o ganho e
mais claro. O tempo caiu de 1.088 s com 1 thread para 0.137 s com 32 threads,
resultando em speedup de 7.94 nessa etapa. Portanto, a paralelizacao trouxe ganho
para o calculo do algoritmo, mas esse ganho ficou mascarado pelo custo da
impressao EPS sequencial.

## Conclusao

A paralelizacao com OpenMP reduziu significativamente o tempo da etapa principal
do K-Means. Porem, o tempo total de execucao teve melhora pequena porque a
funcao `printEPS()` permaneceu sequencial e passou a ser o principal gargalo do
programa. Assim, para avaliar melhor o desempenho da paralelizacao, o tempo da
etapa `kmeans` e mais representativo do que o tempo total com impressao da saida.
