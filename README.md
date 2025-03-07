# Numba.OpenCL

Uma extensão para o Numba que permite usar OpenCL como backend, similar ao módulo `numba.cuda`.

## Visão Geral

O projeto `numba_opencl` é uma extensão que emula a API `numba.cuda` usando PyOpenCL como backend. Isso permite que código existente escrito para GPUs NVIDIA usando `numba.cuda` possa ser facilmente adaptado para execução em uma ampla variedade de dispositivos que suportam OpenCL, incluindo GPUs AMD, Intel, e até mesmo CPUs.

## Recursos Principais

- **API compatível com CUDA**: Use a sintaxe familiar do `numba.cuda` com dispositivos OpenCL
- **Portabilidade**: Execute em qualquer dispositivo compatível com OpenCL (GPUs AMD, Intel, CPUs, etc.)
- **Decoradores JIT**: Compile funções Python para kernels OpenCL com o decorador `@opencl.jit`
- **Gerenciamento de memória**: Transferência de dados entre host e dispositivo, similar ao CUDA
- **Streams**: Suporte a execução assíncrona através de streams
- **Profiling**: Ferramentas para análise de desempenho
- **Funções de sincronização**: Barreiras e operações atômicas

## Instalação

```bash
pip install pyopencl numba numpy siphash24 prettytable
```

Ou instale diretamente da fonte:

```bash
git clone https://github.com/usuario/numba-opencl.git
cd numba-opencl
pip install -e .
```

## Estrutura do Projeto

```
numba_opencl/
  ├── __init__.py     # Exporta os principais componentes
  ├── core.py         # Implementação principal do OpenCL
  ├── decorators.py   # Operações atômicas e sincronização
  ├── utils.py        # Funções utilitárias
  ├── config.py       # Sistema de configuração
  └── profiler.py     # Ferramentas de profiling
main.py               # Exemplos e benchmarks
tests.py              # Testes automatizados
setup.py              # Script de instalação
```

## Uso Básico

```python
import numpy as np
from numba_opencl import ocl

# Definir um kernel OpenCL
@ocl.jit
def vector_add(a, b, c):
    i = ocl.get_global_id(0)
    if i < len(c):
        c[i] = a[i] + b[i]

# Criar arrays
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
c = np.zeros_like(a)

# Transferir para o dispositivo
d_a = ocl.to_device(a)
d_b = ocl.to_device(b)
d_c = ocl.to_device(c)

# Executar kernel (similar ao CUDA)
vector_add(d_a, d_b, d_c, grid=(1,), block=(4,))

# Obter resultado
result = d_c.copy_to_host()
print(result)  # [11, 22, 33, 44]
```

## Exemplos Avançados

### Multiplicação de Matrizes

```python
import numpy as np
from numba_opencl import ocl

@ocl.jit
def matrix_multiply(a, b, c, width):
    row = ocl.get_global_id(0)
    col = ocl.get_global_id(1)

    if row < width and col < width:
        tmp = 0.0
        for i in range(width):
            tmp += a[row * width + i] * b[i * width + col]
        c[row * width + col] = tmp

# Criar matrizes
width = 32
a_mat = np.random.rand(width, width).astype(np.float32)
b_mat = np.random.rand(width, width).astype(np.float32)
c_mat = np.zeros((width, width), dtype=np.float32)

# Reformatar para array unidimensional
a_flat = a_mat.flatten()
b_flat = b_mat.flatten()
c_flat = c_mat.flatten()

# Transferir para o dispositivo
d_a = ocl.to_device(a_flat)
d_b = ocl.to_device(b_flat)
d_c = ocl.to_device(c_flat)

# Executar kernel
block_dim = (16, 16)
grid_dim = (width // block_dim[0] + 1, width // block_dim[1] + 1)
matrix_multiply(d_a, d_b, d_c, width, grid=grid_dim, block=block_dim)

# Obter resultado
result_flat = d_c.copy_to_host()
result_mat = result_flat.reshape((width, width))
```

### Usando Streams para Execução Assíncrona

```python
import numpy as np
from numba_opencl import ocl

@ocl.jit
def vector_add(a, b, c):
    i = ocl.get_global_id(0)
    if i < len(c):
        c[i] = a[i] + b[i]

# Criar streams
stream1 = ocl.stream()
stream2 = ocl.stream()

# Executar kernels em streams diferentes
with stream1:
    vector_add(d_a1, d_b1, d_c1, grid=grid_size, block=block_size)

with stream2:
    vector_add(d_a2, d_b2, d_c2, grid=grid_size, block=block_size)

# Sincronizar
stream1.synchronize()
stream2.synchronize()
```

### Usando o Profiler

```python
from numba_opencl.profiler import profiler

# Ativar profiling
profiler.start()

# Execute seus kernels
vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
matrix_multiply(d_a, d_b, d_c, width, grid=grid_dim, block=block_dim)

# Imprimir estatísticas
profiler.print_stats()

# Parar profiling
profiler.stop()
```

## Gestão de Dispositivos

```python
from numba_opencl import ocl

# Listar dispositivos disponíveis
devices = ocl.list_devices()
for device in devices:
    print(f"ID: {device['id']}, Nome: {device['name']}, Tipo: {device['type']}")

# Selecionar dispositivo específico
ocl.select_device(0)  # Selecionar pelo ID

# Selecionar por tipo
ocl.select_device_by_type('GPU')  # ou 'CPU' ou 'ACCELERATOR'

# Obter informações do dispositivo atual
info = ocl.get_device_info()
print(f"Dispositivo atual: {info['name']} ({info['type']})")
```

## API

### Principais Classes e Funções

#### Configuração e Inicialização

- `ocl.list_devices()`: Lista todos os dispositivos OpenCL disponíveis
- `ocl.select_device(device_id)`: Seleciona um dispositivo pelo ID
- `ocl.select_device_by_type(device_type)`: Seleciona um dispositivo pelo tipo
- `ocl.get_device_info(device_id=None)`: Retorna informações do dispositivo
- `ocl.print_device_info()`: Imprime informações detalhadas do dispositivo atual

#### Compilação e Execução

- `ocl.jit(func)`: Decorator para compilar uma função Python em um kernel OpenCL
- `ocl.synchronize()`: Sincroniza todos os comandos pendentes
- `ocl.device_count()`: Retorna o número de dispositivos OpenCL disponíveis

#### Funções de Grid/Thread

- `ocl.get_global_id(dim)`: Obtém o ID global da thread atual
- `ocl.get_local_id(dim)`: Obtém o ID local da thread
- `ocl.get_group_id(dim)`: Obtém o ID do grupo de trabalho
- `ocl.get_local_size(dim)`: Obtém o tamanho local do grupo
- `ocl.get_global_size(dim)`: Obtém o tamanho global

#### Gerenciamento de Memória

- `ocl.to_device(array)`: Transfere um array NumPy para o dispositivo
- `ocl.device_array(shape, dtype)`: Cria um array vazio no dispositivo
- `ocl.device_array_like(array)`: Cria um array vazio com a mesma forma e tipo
- `ocl.shared_array(shape, dtype)`: Cria um array na memória compartilhada

#### DeviceArray

- `DeviceArray.copy_to_host()`: Copia dados do dispositivo para o host
- `DeviceArray.copy_to_device(array)`: Copia dados do host para o dispositivo
- `DeviceArray.copy_from_device(device_array)`: Copia de outro array do dispositivo
- `DeviceArray.copy_from_device_async(device_array, stream)`: Cópia assíncrona

#### Streams

- `ocl.stream()`: Cria um novo stream
- `stream.synchronize()`: Sincroniza operações no stream
- `stream.wait_event(event)`: Aguarda um evento

#### Profiling

- `profiler.start()`: Inicia o profiling
- `profiler.stop()`: Para o profiling
- `profiler.get_stats(kernel_name=None)`: Obtém estatísticas de execução
- `profiler.print_stats()`: Imprime estatísticas formatadas

### Operações Atômicas e Sincronização

- `atomic_add(array, index, value)`: Adição atômica
- `atomic_max(array, index, value)`: Máximo atômico
- `atomic_min(array, index, value)`: Mínimo atômico
- `atomic_cas(array, index, compare_value, value)`: Compare-and-swap
- `atomic_exch(array, index, value)`: Troca atômica
- `barrier()`: Barreira de sincronização global
- `local_barrier()`: Barreira local dentro de um grupo
- `syncthreads()`: Alias para local_barrier() (compatibilidade CUDA)

## Exemplos Incluídos

O arquivo `main.py` contém vários exemplos:

1. **Soma de Vetores**: Operação básica de elemento a elemento
2. **Multiplicação de Matrizes**: Demonstração de kernel 2D
3. **Filtro de Imagem**: Exemplo de processamento de imagem (blur)
4. **Redução**: Soma de todos os elementos de um array
5. **SAXPY**: Operação Single-Precision A*X Plus Y
6. **Convolução 2D**: Aplicação de filtro Sobel para detecção de bordas

## Benchmarks

O projeto inclui benchmarks que comparam o desempenho do OpenCL versus NumPy puro:

```bash
python main.py --bench
```

Os benchmarks incluem:
- Soma de vetores
- Multiplicação de matrizes
- Redução (soma)
- SAXPY
- Convolução 2D

## Testes

O arquivo `tests.py` contém testes automatizados:

```bash
python tests.py
```

## Compatibilidade

Esta biblioteca foi testada com:
- Python 3.6+
- NumPy 1.20+
- PyOpenCL 2022.1+
- Numba 0.55+

Dispositivos testados:
- GPUs NVIDIA (versões de driver 470+)
- GPUs AMD (drivers ROCm e proprietários)
- GPUs Intel
- CPUs Intel/AMD via OpenCL

## Limitações

- Esta é uma implementação de prova de conceito
- Não é uma integração completa com o compilador Numba
- A tradução Python para OpenCL C é simplificada
- O suporte a tipos complexos é limitado
- Sem suporte para CUDA Unified Memory
- Alguns padrões de programação CUDA avançados não são suportados

## Contribuições

Contribuições são bem-vindas! Por favor, abra um issue ou envie um pull request.

## Licença

MIT