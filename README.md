
# Numba.OpenCL

Uma extensão para o Numba que permite usar OpenCL como backend, similar ao módulo `numba.cuda`.

## Estrutura do Projeto

```
numba_opencl/
  ├── __init__.py     # Exporta os principais componentes
  ├── core.py         # Implementação principal do OpenCL
  ├── decorators.py   # Operações atômicas e sincronização
  └── utils.py        # Funções utilitárias
main.py               # Exemplos e benchmarks
```

## Uso Básico

```python
import numpy as np
from numba_opencl import opencl

# Definir um kernel OpenCL
@opencl.jit
def vector_add(a, b, c):
    i = get_global_id(0)
    if i < c.shape[0]:
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

## Características

- API compatível com `numba.cuda`
- Suporte para dispositivos OpenCL
- Fallback automático para CPU quando OpenCL não está disponível
- Gerenciamento automático de memória
- Compilação just-in-time de kernels

## Limitações

- Esta é uma implementação de prova de conceito
- Não é uma integração completa com o compilador Numba
- Algumas funcionalidades são simuladas
