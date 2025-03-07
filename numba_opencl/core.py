
"""
Módulo principal que implementa a funcionalidade OpenCL
Fornece uma API compatível com numba.cuda para OpenCL
"""

import numpy as np
import pyopencl as cl
import threading
import re
import string
import random
from .utils import inspect_function_source, random_id

# Módulo personalizado numba.opencl
class OpenCLExtension:
    """
    Classe principal que implementa a funcionalidade do OpenCL como uma extensão Numba.
    
    Esta classe oferece uma API similar ao numba.cuda, mas utilizando PyOpenCL como backend.
    Suporta compilação JIT de kernels Python para código OpenCL C, gerenciamento de memória,
    e execução de kernels em dispositivos compatíveis com OpenCL.
    """
    def __init__(self):
        # Inicialização básica do contexto OpenCL
        try:
            self.platforms = cl.get_platforms()
            if not self.platforms:
                raise RuntimeError("Nenhuma plataforma OpenCL encontrada")
            
            self.devices = self.platforms[0].get_devices(device_type=cl.device_type.ALL)
            if not self.devices:
                raise RuntimeError("Nenhum dispositivo OpenCL encontrado")
            
            self.device = self.devices[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self._local_size = 64  # Tamanho padrão do grupo local
            self._stream = None
            self._thread_local = threading.local()
            
            # Cache para programas OpenCL compilados
            self._program_cache = {}
            
            # Informações do dispositivo
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            self.max_work_item_dimensions = self.device.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
            self.max_work_item_sizes = self.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
            
            print(f"OpenCL Inicializado: {self.device.name}")
            print(f"Max Group Size: {self.max_work_group_size}")
            print(f"Max Work Item Dimensions: {self.max_work_item_dimensions}")
            print(f"Max Work Item Sizes: {self.max_work_item_sizes}")
            
        except Exception as e:
            print(f"Erro ao inicializar OpenCL: {e}")
            # Fallback para CPU (simulação)
            self.device = None
            self.context = None
            self.queue = None
            print("Usando fallback para CPU")

    @property
    def local_size(self):
        return self._local_size
    
    @local_size.setter
    def local_size(self, size):
        self._local_size = size

    # Compilar código OpenCL
    def _compile_opencl_code(self, kernel_code, kernel_name):
        """Compila um kernel OpenCL e retorna o programa compilado"""
        if kernel_name in self._program_cache:
            return self._program_cache[kernel_name]
        
        try:
            program = cl.Program(self.context, kernel_code).build()
            self._program_cache[kernel_name] = program
            return program
        except cl.RuntimeError as e:
            print(f"Erro ao compilar kernel OpenCL: {e}")
            print(f"Código do kernel:\n{kernel_code}")
            raise

    # Gerador de código OpenCL a partir do Python
    def _generate_opencl_code(self, func, arg_types=None):
        """Gera código OpenCL a partir de uma função Python"""
        func_src = inspect_function_source(func)
        func_name = func.__name__
        
        # Processamento básico do código da função
        # Isso é uma simplificação, uma implementação real analisaria a AST 
        # e converteria construções Python para OpenCL C
        
        # Substituições simples para tipos Python->OpenCL
        replacements = {
            'np.float32': 'float',
            'np.float64': 'double',
            'np.int32': 'int',
            'np.int64': 'long',
            'numba.float32': 'float',
            'numba.float64': 'double',
            'numba.int32': 'int',
            'numba.int64': 'long',
        }
        
        # Substituir tipos Python por tipos OpenCL
        for py_type, cl_type in replacements.items():
            func_src = func_src.replace(py_type, cl_type)
        
        # Substituir acessos a arrays por acessos a matrizes OpenCL
        # Isso é muito simplificado
        func_src = re.sub(r'(\w+)\[(\w+)\]', r'\1[\2]', func_src)
        
        # Criar template para o kernel OpenCL
        kernel_template = """
        __kernel void {func_name}({params}) {{
            int gid = get_global_id(0);
            {func_body}
        }}
        """
        
        # Extrair corpo da função
        body_lines = []
        in_body = False
        for line in func_src.split('\n'):
            line = line.strip()
            if line.startswith('def '):
                in_body = True
                continue
            if in_body and line:
                # Remover indentação
                if line.startswith('    '):
                    line = line[4:]
                body_lines.append(line)
        
        func_body = '\n    '.join(body_lines)
        
        # Criar parâmetros do kernel
        # Em uma implementação real, isso seria baseado na análise do tipo
        if arg_types:
            params = []
            for i, arg_type in enumerate(arg_types):
                if arg_type == 'array':
                    params.append(f"__global float* arg{i}")
                else:
                    params.append(f"const {arg_type} arg{i}")
            params_str = ", ".join(params)
        else:
            # Fallback simples quando os tipos não são fornecidos
            params_str = ", ".join([f"__global float* arg{i}" for i in range(5)])
        
        kernel_code = kernel_template.format(
            func_name=func_name,
            params=params_str,
            func_body=func_body
        )
        
        return kernel_code, func_name

    # Simulando comportamento similar ao numba.cuda.jit
    def jit(self, func=None, device=False, **kwargs):
        def decorator(func):
            # Gerar ID único para a função
            func_id = f"{func.__name__}_{random_id()}"
            
            # Função wrapper que será retornada
            def kernel_wrapper(*args, **kwargs):
                if self.context is None:
                    # Fallback para CPU se OpenCL não estiver disponível
                    print("OpenCL não disponível, executando na CPU")
                    return func(*args)
                
                # Analisar argumentos
                kernel_args = []
                kernel_arg_types = []
                device_arrays = []
                
                for i, arg in enumerate(args):
                    if isinstance(arg, DeviceArray):
                        # Já é um array no dispositivo
                        kernel_args.append(arg.cl_buffer)
                        kernel_arg_types.append('array')
                        device_arrays.append(arg)
                    elif isinstance(arg, np.ndarray):
                        # Criar buffer OpenCL
                        cl_buffer = cl.Buffer(self.context, 
                                             cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                             hostbuf=arg)
                        kernel_args.append(cl_buffer)
                        kernel_arg_types.append('array')
                        # Criar DeviceArray para rastrear o buffer
                        device_arrays.append(DeviceArray(cl_buffer, arg.shape, arg.dtype, self))
                    else:
                        # Valor escalar
                        kernel_args.append(arg)
                        # Determinar tipo OpenCL baseado no tipo Python
                        if isinstance(arg, int):
                            kernel_arg_types.append('int')
                        elif isinstance(arg, float):
                            kernel_arg_types.append('float')
                        else:
                            kernel_arg_types.append('void*')  # Tipo genérico
                
                # Configurar tamanho da grade/bloco
                grid_dim = kwargs.get('grid', (1,))
                block_dim = kwargs.get('block', (self._local_size,))
                
                if isinstance(grid_dim, int):
                    grid_dim = (grid_dim,)
                if isinstance(block_dim, int):
                    block_dim = (block_dim,)
                
                # Calcular dimensões globais e locais para OpenCL
                global_size = tuple(g * b for g, b in zip(grid_dim, block_dim))
                local_size = block_dim
                
                try:
                    # Gerar código do kernel
                    kernel_code, kernel_name = self._generate_opencl_code(func, kernel_arg_types)
                    
                    # Compilar kernel
                    program = self._compile_opencl_code(kernel_code, func_id)
                    
                    # Obter referência ao kernel
                    kernel = getattr(program, kernel_name)
                    
                    # Configurar kernel
                    kernel.set_args(*kernel_args)
                    
                    # Executar kernel
                    event = cl.enqueue_nd_range_kernel(
                        self.queue, kernel, global_size, local_size)
                    
                    # Aguardar conclusão
                    event.wait()
                    
                    print(f"Kernel '{func.__name__}' executado com sucesso")
                    print(f"Dimensões globais: {global_size}, locais: {local_size}")
                    
                    # Retornar os arrays do dispositivo
                    if len(device_arrays) == 1:
                        return device_arrays[0]
                    elif len(device_arrays) > 1:
                        return tuple(device_arrays)
                    
                except Exception as e:
                    print(f"Erro na execução do kernel: {e}")
                    # Fallback para CPU
                    print("Executando na CPU como fallback")
                    return func(*args)
            
            return kernel_wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    # Mapeamento de array similar ao numba.cuda
    def to_device(self, arr):
        """Transfere um array NumPy para o dispositivo OpenCL"""
        if self.context is None:
            print("OpenCL não disponível, usando array NumPy")
            return arr
            
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        
        try:
            cl_buffer = cl.Buffer(self.context, 
                                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                hostbuf=arr)
            return DeviceArray(cl_buffer, arr.shape, arr.dtype, self)
        except Exception as e:
            print(f"Erro ao transferir para o dispositivo: {e}")
            return arr
    
    # Gerenciamento de stream similar ao numba.cuda
    def stream(self):
        """Cria uma nova fila de comandos (equivalente a stream no CUDA)"""
        if self.context is None:
            return None
            
        if self._stream is None:
            self._stream = cl.CommandQueue(self.context)
        return self._stream

    # Gerenciamento de memória compartilhada
    def shared_array(self, shape, dtype):
        """Cria um array na memória compartilhada"""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        return SharedMemory(size, shape, dtype)
        
    # Métodos adicionais como no CUDA
    
    def synchronize(self):
        """Sincroniza todos os comandos pendentes"""
        if self.queue is not None:
            self.queue.finish()
    
    def device_count(self):
        """Retorna o número de dispositivos OpenCL disponíveis"""
        if not self.platforms:
            return 0
        count = 0
        for platform in self.platforms:
            count += len(platform.get_devices())
        return count
    
    def get_device(self, device_id=0):
        """Obtém um dispositivo específico pelo ID"""
        if not self.platforms:
            return None
            
        count = 0
        for platform in self.platforms:
            devices = platform.get_devices()
            if device_id < count + len(devices):
                return devices[device_id - count]
            count += len(devices)
        return None
    
    def select_device(self, device_id=0):
        """Seleciona um dispositivo para usar"""
        device = self.get_device(device_id)
        if device:
            self.device = device
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self._program_cache = {}  # Limpar cache ao mudar de dispositivo
            print(f"Dispositivo selecionado: {self.device.name}")
        else:
            print(f"Dispositivo {device_id} não encontrado")
    
    def get_current_device(self):
        """Retorna o dispositivo atual"""
        return self.device
    
    # Funções de Grid/Block como no CUDA
    def grid(self, x=1, y=1, z=1):
        """Define uma grade de blocos"""
        return (x, y, z)
    
    def block(self, x=1, y=1, z=1):
        """Define um bloco de threads"""
        return (x, y, z)


# Classe para representar arrays no dispositivo
class DeviceArray:
    def __init__(self, cl_buffer, shape, dtype, opencl_ext):
        self.cl_buffer = cl_buffer
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(shape)
        self.opencl_ext = opencl_ext
        self.ndim = len(shape) if isinstance(shape, tuple) else 1
    
    def copy_to_host(self):
        """Copia dados do dispositivo para o host"""
        if self.cl_buffer is None:
            return None
            
        try:
            result = np.empty(self.shape, dtype=self.dtype)
            cl.enqueue_copy(self.opencl_ext.queue, result, self.cl_buffer)
            self.opencl_ext.queue.finish()  # Garantir que a cópia seja concluída
            return result
        except Exception as e:
            print(f"Erro ao copiar para o host: {e}")
            return None
    
    def copy_to_device(self, arr):
        """Copia dados do host para o dispositivo"""
        if self.cl_buffer is None:
            return self
            
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=self.dtype)
        
        if arr.shape != self.shape:
            raise ValueError(f"Forma incompatível: {arr.shape} vs {self.shape}")
        
        try:
            cl.enqueue_copy(self.opencl_ext.queue, self.cl_buffer, arr)
            self.opencl_ext.queue.finish()  # Garantir que a cópia seja concluída
            return self
        except Exception as e:
            print(f"Erro ao copiar para o dispositivo: {e}")
            return self
    
    def __getitem__(self, key):
        """Suporte a indexação"""
        # Implementação simplificada
        # Uma implementação completa gerenciaria a transferência parcial
        host_array = self.copy_to_host()
        return host_array[key]
    
    def __setitem__(self, key, value):
        """Suporte a atribuição por indexação"""
        # Implementação simplificada
        # Uma implementação completa gerenciaria a transferência parcial
        host_array = self.copy_to_host()
        host_array[key] = value
        self.copy_to_device(host_array)
    
    def __len__(self):
        """Suporte ao operador len()"""
        if isinstance(self.shape, tuple) and len(self.shape) > 0:
            return self.shape[0]
        return 0
    
    def __str__(self):
        """Representação em string"""
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype})"
    
    def __repr__(self):
        """Representação oficial"""
        return self.__str__()


# Classe para memória compartilhada
class SharedMemory:
    def __init__(self, size, shape, dtype):
        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.data = np.zeros(shape, dtype=dtype)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value


# Funções auxiliares do OpenCL
def get_global_id(dim):
    """Emula a função get_global_id do OpenCL"""
    import pyopencl as cl
    try:
        return cl.get_global_id(dim)
    except:
        return 0  # Fallback

# Criar instância global do módulo opencl
opencl = OpenCLExtension()
# Adicionar get_global_id como método do opencl
opencl.get_global_id = get_global_id
