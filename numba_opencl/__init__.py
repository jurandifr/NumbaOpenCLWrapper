
from .core import opencl, DeviceArray, SharedMemory
from .decorators import atomic_add, atomic_max, atomic_min, syncthreads

__all__ = [
    'opencl',
    'DeviceArray',
    'SharedMemory',
    'atomic_add',
    'atomic_max',
    'atomic_min',
    'syncthreads'
]
"""
numba_opencl: Uma extensão para o Numba que permite usar OpenCL como backend.

Este pacote fornece uma API compatível com numba.cuda para usar OpenCL,
permitindo que código existente seja facilmente adaptado para execução
em uma variedade maior de aceleradores de hardware.

Exemplo básico:
---------------
```python
import numpy as np
from numba_opencl import opencl

@opencl.jit
def add(a, b, c):
    i = opencl.get_global_id(0)
    if i < len(c):
        c[i] = a[i] + b[i]

# Dados de entrada
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
c = np.zeros_like(a)

# Transferir para GPU
d_a = opencl.to_device(a)
d_b = opencl.to_device(b)
d_c = opencl.to_device(c)

# Executar kernel
add(d_a, d_b, d_c, grid=(1,), block=(4,))

# Obter resultado
result = d_c.copy_to_host()
print(result)  # [11, 22, 33, 44]
```
"""

from .core import opencl, get_global_id
from .decorators import barrier, atomic_add, local_barrier

__version__ = '0.1.0'

# Exportar principais componentes
__all__ = [
    'opencl',         # Instância principal do módulo
    'get_global_id',  # Função para obter ID global
    'barrier',        # Barreira de sincronização
    'atomic_add',     # Operação atômica de adição
    'local_barrier',  # Barreira local (similar a __syncthreads() do CUDA)
]
