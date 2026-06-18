# Projeto PPL - K-Means

Este projeto compara tres implementacoes do algoritmo K-Means:

- `src/Kmeans.c`: versao sequencial em CPU.
- `src/Kmeans_cpu.c`: versao paralela em CPU com OpenMP.
- `src/Kmeans_gpu_openmp.c`: versao paralela em GPU com OpenMP target/offload.
- `src/Kmeans_cuda.cu`: versao paralela em GPU com CUDA.

O `main` das versoes executa `test2()`, que gera 2.000.000 observacoes,
usa `k = 11` clusters e imprime a visualizacao em formato EPS no `stdout`.

## Estrutura do repositorio

```text
.
├── README.md
├── src/
│   ├── Kmeans.c
│   ├── Kmeans_cpu.c
│   └── Kmeans_cuda.cu
├── benchmarks/
│   ├── benchmarks_cuda.txt
│   └── relatorio_kmeans_benchmarks.md
├── outputs/
│   └── saida_cuda.eps
└── bin/
    ├── Kmeans.exe
    ├── Kmeans_cpu.exe
    ├── Kmeans_cpu_omp.exe
    ├── Kmeans_cpu_omp
    ├── Kmeans_cuda
    └── Kmeans_seq
```

`bin/` guarda executaveis ja compilados. `outputs/` guarda saidas geradas, como
arquivos EPS. Novos binarios e novas saidas sao ignorados pelo Git.

## Compilacao

No WSL/Linux, entre na pasta do projeto:

```bash
cd "/mnt/c/Users/Victor Souza Lima/Documents/Projeto-PPL"
```

Crie a pasta de binarios, caso ela nao exista:

```bash
mkdir -p bin outputs
```

Versao sequencial:

```bash
gcc -O0 src/Kmeans.c -o bin/Kmeans_seq_O0 -lm
```

Versao OpenMP:

```bash
gcc -O0 -fopenmp src/Kmeans_cpu.c -o bin/Kmeans_cpu_omp -lm
```

Versao OpenMP com offload para GPU:

```bash
gcc -O2 -fopenmp -foffload=nvptx-none src/Kmeans_gpu_openmp.c -o bin/Kmeans_gpu_openmp -lm
```

Se o compilador nao tiver um dispositivo OpenMP configurado, as regioes
`target` podem executar no host como fallback. Para confirmar se o runtime
encontrou GPU, veja no benchmark impresso em `stderr` o campo `devices`.

Versao CUDA:

```bash
nvcc -O0 -arch=sm_89 src/Kmeans_cuda.cu -o bin/Kmeans_cuda
```

Se o `nvcc` nao reconhecer `sm_89`, use:

```bash
nvcc -O2 -arch=sm_86 src/Kmeans_cuda.cu -o bin/Kmeans_cuda
```

## Execucao

Como o programa imprime uma imagem EPS com cerca de 5.000.000 pontos, recomenda-se
redirecionar a saida para arquivo:

```bash
time ./bin/Kmeans_seq > outputs/saida_seq.eps
time ./bin/Kmeans_cpu_omp > outputs/saida_cpu_omp.eps
time ./bin/Kmeans_gpu_openmp > outputs/saida_gpu_openmp.eps
time ./bin/Kmeans_cuda > outputs/saida_cuda.eps
```

Rodar sem redirecionamento imprime o EPS inteiro no terminal, o que costuma
dominar o tempo total e dificultar a comparacao.

Para medir o tempo sem salvar a saida EPS em arquivo, redirecione o `stdout`
para `/dev/null`:

```bash
time ./bin/Kmeans_seq > /dev/null
time ./bin/Kmeans_cpu_omp > /dev/null
time ./bin/Kmeans_gpu_openmp > /dev/null
time ./bin/Kmeans_cuda > /dev/null
```

Esse modo evita gravar um arquivo gigante em disco. Ainda assim, a funcao
`printEPS()` continua sendo executada e formatando a saida; ela apenas e
descartada pelo sistema operacional.

Para verificar o impacto das otimizacoes do compilador na versao sequencial,
compile e execute duas versoes:

```bash
gcc -O0 src/Kmeans.c -o bin/Kmeans_seq_O0 -lm
gcc -O2 src/Kmeans.c -o bin/Kmeans_seq_O2 -lm

time ./bin/Kmeans_seq_O0 > /dev/null
time ./bin/Kmeans_seq_O2 > /dev/null
```

`-O0` desativa as otimizacoes principais do GCC. `-O2` ativa otimizacoes comuns
de desempenho, como simplificacoes de loops e melhor uso de registradores.

## Benchmarks

Os resultados e explicacoes estao em:

- `benchmarks/benchmarks_cuda.txt`: anotacoes do benchmark CUDA e explicacao do
  arquivo EPS.
- `benchmarks/relatorio_kmeans_benchmarks.md`: relatorio consolidado comparando
  sequencial, OpenMP e CUDA.

Resultado CUDA registrado:

```text
Benchmark CUDA: size=1000000 k=11 block=256
geracao_dados: 0.077748 s
kmeans_cuda: 0.309163 s
printEPS: 4.220437 s
total_medido: 4.607348 s
```

O benchmark mostra que, mesmo com o K-Means em CUDA, a funcao `printEPS()`
continua sendo o principal custo quando a saida EPS completa e gerada.

Com `size = 2000000`, a saida EPS completa ainda pode ficar grande e
demorar bastante para ser gravada. Para medir apenas o algoritmo, considere
temporariamente comentar a chamada de `printEPS()` ou usar a funcao `test()`,
que gera uma entrada menor.

## O que o EPS contem

O arquivo EPS e uma imagem vetorial dos clusters:

- cada linha `R G B setrgbcolor` define a cor de um cluster;
- cada linha `x y c` desenha uma observacao;
- cada linha `0 setgray x y s` desenha o centroide;
- `%%EOF` indica o fim do arquivo.

As coordenadas impressas ja estao escaladas para caber em uma imagem de 400x400,
nao sendo as coordenadas originais dos pontos.

## Observacoes sobre Git

O `.gitignore` ignora novos executaveis, objetos de compilacao e saidas geradas.
Alguns executaveis antigos ja estavam versionados e foram movidos para `bin/`.
Se a intencao for remover esses binarios do historico de arquivos rastreados,
use depois:

```bash
git rm --cached bin/Kmeans.exe bin/Kmeans_cpu.exe bin/Kmeans_cpu_omp.exe
```

Depois disso, novos builds permanecerao fora do Git por causa do `.gitignore`.
