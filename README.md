
# Numba.OpenCL

Uma extensão para o Numba que permite usar OpenCL como backend, similar ao módulo `numba.cuda`.

## Visão Geral

O projeto `numba_opencl` é uma extensão que emula a API `numba.cuda` usando PyOpenCL como backend. Isso permite que código existente escrito para GPUs NVIDIA usando `numba.cuda` possa ser facilmente adaptado para execução em uma ampla variedade de dispositivos que suportam OpenCL, incluindo GPUs AMD, Intel, e até mesmo CPUs.

## Instalação

```bash
pip install pyopencl numba numpy siphash24
```

## Estrutura do Projeto

```
numba_opencl/
  ├── __init__.py     # Exporta os principais componentes
  ├── core.py         # Implementação principal do OpenCL
  ├── decorators.py   # Operações atômicas e sincronização
  └── utils.py        # Funções utilitárias
main.py               # Exemplos e benchmarks
tests.py              # Testes automatizados
```

## Uso Básico

```python
import numpy as np
from numba_opencl import opencl

# Definir um kernel OpenCL
@opencl.jit
def vector_add(a, b, c):
    i = opencl.get_global_id(0)
    if i < len(c):
        c[i] = a[i] + b[i]

# Criar arrays
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
c = np.zeros_like(a)

# Transferir para o dispositivo
d_a = opencl.to_device(a)
d_b = opencl.to_device(b)
d_c = opencl.to_device(c)

# Executar kernel (similar ao CUDA)
vector_add(d_a, d_b, d_c, grid=(1,), block=(4,))

# Obter resultado
result = d_c.copy_to_host()
print(result)  # [11, 22, 33, 44]
```

## API

### Principais Funções

- `opencl.jit(func)`: Decorator para compilar uma função Python em um kernel OpenCL
- `opencl.to_device(array)`: Transfere um array NumPy para o dispositivo
- `opencl.synchronize()`: Sincroniza todos os comandos pendentes
- `opencl.device_count()`: Retorna o número de dispositivos OpenCL disponíveis
- `opencl.select_device(device_id)`: Seleciona um dispositivo específico
- `opencl.get_global_id(dim)`: Obtém o ID global da thread atual

### Gestão de Memória

- `DeviceArray.copy_to_host()`: Copia dados do dispositivo para o host
- `DeviceArray.copy_to_device(array)`: Copia dados do host para o dispositivo
- `opencl.shared_array(shape, dtype)`: Cria um array na memória compartilhada

### Sincronização

- `barrier()`: Implementa uma barreira de sincronização
- `local_barrier()`: Sincroniza threads dentro de um grupo de trabalho (similar ao `__syncthreads()` do CUDA)

## Exemplos

O arquivo `main.py` contém vários exemplos:

1. **Soma de Vetores**: Operação básica de elemento a elemento
2. **Multiplicação de Matrizes**: Demonstração de kernel 2D
3. **Filtro de Imagem**: Exemplo de processamento de imagem (blur)

## Benchmarks

O projeto inclui benchmarks que comparam o desempenho do OpenCL versus NumPy puro:

```python
# Executar benchmarks
python main.py
```

## Testes

O arquivo `tests.py` contém testes automatizados:

```python
# Executar testes
python tests.py
```

## Limitações

- Esta é uma implementação de prova de conceito
- Não é uma integração completa com o compilador Numba
- A tradução Python para OpenCL C é simplificada
- O suporte a tipos complexos é limitado

## Licença

MIT

## Contribuições

Contribuições são bem-vindas! Por favor, abra um issue ou envie um pull request.
