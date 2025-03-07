
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

Seleção de dispositivo:
----------------------
```python
# Listar todos os dispositivos disponíveis
devices = opencl.list_devices()
for device in devices:
    print(f"{device['id']}: {device['name']} ({device['type']})")

# Selecionar um dispositivo específico pelo ID
opencl.select_device(1)

# Ou selecionar por tipo
opencl.select_device_by_type('GPU')  # Ou 'CPU', 'ACCELERATOR'
```
"""

from .core import (
    opencl, 
    get_global_id, 
    get_local_id, 
    get_group_id, 
    get_local_size, 
    get_global_size,
    barrier,
    DeviceArray,
    SharedMemory,
    Stream,
    Event
)

from .decorators import (
    atomic_add, 
    atomic_max, 
    atomic_min,
    atomic_cas,
    atomic_exch,
    barrier,
    local_barrier,
    syncthreads,
    mem_fence,
    workgroup,
    shfl,
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE
)

from .utils import (
    check_opencl_support,
    compare_arrays,
    format_size
)

__version__ = '0.2.0'

# Exportar principais componentes
__all__ = [
    # Core
    'opencl',           # Instância principal do módulo
    'get_global_id',    # Função para obter ID global
    'get_local_id',     # Função para obter ID local
    'get_group_id',     # Função para obter ID do grupo
    'get_local_size',   # Função para obter tamanho local
    'get_global_size',  # Função para obter tamanho global
    'barrier',          # Barreira de sincronização
    'DeviceArray',      # Classe para arrays no dispositivo
    'SharedMemory',     # Classe para memória compartilhada
    'Stream',           # Classe para streams (filas de comando)
    'Event',            # Classe para eventos
    
    # Decoradores
    'atomic_add',       # Operação atômica de adição
    'atomic_max',       # Operação atômica de máximo
    'atomic_min',       # Operação atômica de mínimo
    'atomic_cas',       # Operação atômica de comparação e troca
    'atomic_exch',      # Operação atômica de troca
    'local_barrier',    # Barreira local (similar a __syncthreads)
    'syncthreads',      # Alias para local_barrier
    'mem_fence',        # Barreira de memória
    'workgroup',        # Classe para operações de grupo de trabalho
    'shfl',             # Classe para operações de shuffle
    'CLK_LOCAL_MEM_FENCE',  # Constante para barreira local
    'CLK_GLOBAL_MEM_FENCE', # Constante para barreira global
    
    # Utilidades
    'check_opencl_support',  # Verificar suporte OpenCL
    'compare_arrays',       # Comparar arrays com tolerância
    'format_size',          # Formatar tamanho em bytes
]

# Configuração baseada em variáveis de ambiente
from .utils import set_default_device_from_env
set_default_device_from_env(opencl)
