
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
import importlib.util
import os
import warnings
from .utils import inspect_function_source, random_id, check_opencl_support

# Verificar se siphash24 está disponível
SIPHASH_AVAILABLE = importlib.util.find_spec("siphash24") is not None
if not SIPHASH_AVAILABLE:
    warnings.warn("Pacote siphash24 não encontrado. Usando alternativa menos eficiente para hash.")

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
        from .utils import check_opencl_support
        
        # Variáveis para fallback
        self._local_size = 64  # Tamanho padrão do grupo local
        self._streams = {}     # Dicionário de streams (filas de comandos)
        self._active_stream = None
        self._thread_local = threading.local()
        self._program_cache = {}
        self._current_device_id = 0
        self._available_devices = []
        self._device_info_cache = None
        
        # Verificar suporte OpenCL
        self.opencl_available, device_info = check_opencl_support()
        self._device_info_cache = device_info
        
        # Inicialização de dispositivos
        self._init_devices()
        
    def _init_devices(self):
        """Inicializa dispositivos OpenCL disponíveis"""
        if not self.opencl_available:
            print("OpenCL não disponível. Usando fallback para CPU (simulação)")
            # Fallback para CPU (simulação)
            self.platforms = []
            self.devices = []
            self.device = None
            self.context = None
            self.queue = None
            self.max_work_group_size = 256
            self.max_work_item_dimensions = 3
            self.max_work_item_sizes = (256, 256, 256)
            self._current_device_id = -1
            return
            
        try:
            self.platforms = cl.get_platforms()
            if not self.platforms:
                raise RuntimeError("Nenhuma plataforma OpenCL encontrada")
            
            # Coletar todos os dispositivos de todas as plataformas
            self._available_devices = []
            for platform in self.platforms:
                for device_type in [cl.device_type.GPU, cl.device_type.CPU, cl.device_type.ACCELERATOR, cl.device_type.ALL]:
                    try:
                        devices = platform.get_devices(device_type=device_type)
                        if devices:
                            for device in devices:
                                self._available_devices.append((platform, device))
                    except:
                        pass
            
            if not self._available_devices:
                raise RuntimeError("Nenhum dispositivo OpenCL encontrado")
            
            # Selecionar primeiro dispositivo por padrão
            self.select_device(0)
            
        except Exception as e:
            print(f"Erro ao inicializar OpenCL: {e}")
            # Fallback para CPU (simulação)
            self.device = None
            self.context = None
            self.queue = None
            self.max_work_group_size = 256
            self.max_work_item_dimensions = 3
            self.max_work_item_sizes = (256, 256, 256)
            self._current_device_id = -1
            print("Usando fallback para CPU (simulação)")

    @property
    def local_size(self):
        return self._local_size
    
    @local_size.setter
    def local_size(self, size):
        self._local_size = size

    def list_devices(self):
        """Lista todos os dispositivos OpenCL disponíveis"""
        if not self.opencl_available:
            print("OpenCL não disponível neste sistema")
            return []
            
        devices = []
        for i, (platform, device) in enumerate(self._available_devices):
            device_name = device.name
            device_type = cl.device_type.to_string(device.type)
            platform_name = platform.name
            compute_units = device.max_compute_units
            
            devices.append({
                'id': i,
                'name': device_name,
                'type': device_type,
                'platform': platform_name,
                'compute_units': compute_units,
                'current': (i == self._current_device_id)
            })
            
        return devices

    def get_device_count(self):
        """Retorna o número de dispositivos OpenCL disponíveis"""
        return len(self._available_devices)

    def get_device_info(self, device_id=None):
        """Retorna informações detalhadas sobre um dispositivo"""
        if device_id is None:
            device_id = self._current_device_id
            
        if device_id < 0 or device_id >= len(self._available_devices):
            return None
            
        platform, device = self._available_devices[device_id]
        
        return {
            'id': device_id,
            'name': device.name,
            'type': cl.device_type.to_string(device.type),
            'platform': platform.name,
            'vendor': device.vendor,
            'version': device.version,
            'driver_version': device.driver_version,
            'compute_units': device.max_compute_units,
            'max_work_group_size': device.max_work_group_size,
            'max_work_item_sizes': device.max_work_item_sizes,
            'global_mem_size': device.global_mem_size,
            'local_mem_size': device.local_mem_size,
            'current': (device_id == self._current_device_id)
        }

    def select_device(self, device_id):
        """Seleciona um dispositivo OpenCL pelo ID"""
        if not self.opencl_available or not self._available_devices:
            self._current_device_id = -1
            print("OpenCL não disponível. Usando modo de simulação CPU.")
            return False
            
        if device_id < 0 or device_id >= len(self._available_devices):
            print(f"ID de dispositivo inválido: {device_id}. Dispositivos disponíveis: 0-{len(self._available_devices)-1}")
            return False
            
        # Selecionar dispositivo
        platform, device = self._available_devices[device_id]
        
        try:
            # Criar novo contexto e fila
            self.device = device
            self.platform = platform
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)
            
            # Atualizar streams
            self._streams = {}
            self._active_stream = None
            
            # Limpar caches associados ao dispositivo anterior
            self._program_cache = {}
            
            # Obter informações do dispositivo
            self.max_work_group_size = device.max_work_group_size
            self.max_work_item_dimensions = device.max_work_item_dimensions
            self.max_work_item_sizes = device.max_work_item_sizes
            
            # Atualizar ID do dispositivo atual
            self._current_device_id = device_id
            
            print(f"Dispositivo selecionado: {device.name}")
            print(f"Plataforma: {platform.name}")
            print(f"Tipo: {cl.device_type.to_string(device.type)}")
            print(f"Unidades de computação: {device.max_compute_units}")
            
            return True
            
        except Exception as e:
            print(f"Erro ao selecionar dispositivo: {e}")
            return False

    def select_device_by_type(self, device_type='GPU'):
        """Seleciona um dispositivo pelo tipo (GPU, CPU, ACCELERATOR)"""
        if not self.opencl_available:
            return False
            
        # Mapear string para tipo de dispositivo OpenCL
        type_map = {
            'GPU': cl.device_type.GPU,
            'CPU': cl.device_type.CPU,
            'ACCELERATOR': cl.device_type.ACCELERATOR
        }
        
        cl_device_type = type_map.get(device_type.upper(), None)
        if cl_device_type is None:
            print(f"Tipo de dispositivo inválido: {device_type}. Tipos válidos: GPU, CPU, ACCELERATOR")
            return False
            
        # Procurar dispositivo do tipo especificado
        for i, (platform, device) in enumerate(self._available_devices):
            if device.type == cl_device_type:
                return self.select_device(i)
                
        print(f"Nenhum dispositivo do tipo {device_type} encontrado")
        return False

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
        
        # Processamento do código da função
        # Substituições de tipos Python->OpenCL
        replacements = {
            'np.float32': 'float',
            'np.float64': 'double',
            'np.int32': 'int',
            'np.int64': 'long',
            'numba.float32': 'float',
            'numba.float64': 'double',
            'numba.int32': 'int',
            'numba.int64': 'long',
            'opencl.get_global_id': 'get_global_id',
            'opencl.get_local_id': 'get_local_id',
            'opencl.get_group_id': 'get_group_id',
            'opencl.get_local_size': 'get_local_size',
            'opencl.get_global_size': 'get_global_size',
            'opencl.barrier': 'barrier',
            'barrier()': 'barrier(CLK_GLOBAL_MEM_FENCE)',
            'math.sin': 'sin',
            'math.cos': 'cos',
            'math.exp': 'exp',
            'math.log': 'log',
            'math.sqrt': 'sqrt',
        }
        
        # Substituir tipos Python por tipos OpenCL
        for py_type, cl_type in replacements.items():
            func_src = func_src.replace(py_type, cl_type)
        
        # Substituir acessos a arrays por acessos a matrizes OpenCL
        # Isso é simplificado; uma implementação real faria análise AST
        func_src = re.sub(r'(\w+)\[(\w+)\]', r'\1[\2]', func_src)
        
        # Criar template para o kernel OpenCL
        kernel_template = """
        __kernel void {func_name}({params}) {{
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
                
                # Completar dimensões para até 3D
                grid_dim = grid_dim + (1,) * (3 - len(grid_dim))
                block_dim = block_dim + (1,) * (3 - len(block_dim))
                
                # Calcular dimensões globais e locais para OpenCL
                global_size = tuple(g * b for g, b in zip(grid_dim, block_dim))
                local_size = block_dim
                
                # Verificar limites do dispositivo
                if any(ls > self.max_work_group_size for ls in local_size):
                    print(f"Aviso: Tamanho do grupo de trabalho {local_size} excede o máximo do dispositivo {self.max_work_group_size}")
                    # Ajustar tamanho local para não exceder o máximo
                    scale = self.max_work_group_size / max(local_size)
                    local_size = tuple(int(ls * scale) for ls in local_size)
                    print(f"Ajustado para: {local_size}")
                
                try:
                    # Gerar código do kernel
                    kernel_code, kernel_name = self._generate_opencl_code(func, kernel_arg_types)
                    
                    # Compilar kernel
                    program = self._compile_opencl_code(kernel_code, func_id)
                    
                    # Obter referência ao kernel
                    kernel = getattr(program, kernel_name)
                    
                    # Configurar kernel
                    kernel.set_args(*kernel_args)
                    
                    # Usar stream ativo se disponível
                    queue = self._active_stream if self._active_stream else self.queue
                    
                    # Executar kernel
                    event = cl.enqueue_nd_range_kernel(
                        queue, kernel, global_size, local_size)
                    
                    # Aguardar conclusão
                    event.wait()
                    
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
            
        # Criar nova fila de comandos
        new_stream = cl.CommandQueue(self.context)
        stream_id = id(new_stream)
        self._streams[stream_id] = new_stream
        return Stream(new_stream, stream_id, self)
    
    def get_current_stream(self):
        """Retorna o stream atual"""
        return self._active_stream if self._active_stream else self.queue

    def set_current_stream(self, stream):
        """Define o stream atual"""
        if isinstance(stream, Stream):
            self._active_stream = stream.queue
        elif stream is None:
            self._active_stream = None
        else:
            self._active_stream = stream

    # Gerenciamento de memória compartilhada
    def shared_array(self, shape, dtype):
        """Cria um array na memória compartilhada"""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        return SharedMemory(size, shape, dtype)
    
    # Alocação de memória sem inicialização
    def device_array(self, shape, dtype=np.float32):
        """Aloca um array no dispositivo sem inicializá-lo"""
        if self.context is None:
            return np.empty(shape, dtype=dtype)
        
        try:
            size = np.prod(shape) * np.dtype(dtype).itemsize
            cl_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
            return DeviceArray(cl_buffer, shape, dtype, self)
        except Exception as e:
            print(f"Erro ao alocar array no dispositivo: {e}")
            return np.empty(shape, dtype=dtype)
    
    def device_array_like(self, arr):
        """Aloca um array no dispositivo com a mesma forma e tipo que arr"""
        return self.device_array(arr.shape, arr.dtype)
        
    # Métodos adicionais como no CUDA
    
    def synchronize(self):
        """Sincroniza todos os comandos pendentes"""
        if self.queue is not None:
            self.queue.finish()
        
        # Sincronizar todos os streams
        for stream in self._streams.values():
            stream.finish()
    
    def device_count(self):
        """Retorna o número de dispositivos OpenCL disponíveis"""
        return len(self._available_devices)
    
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
    
    # Funções OpenCL equivalentes às do CUDA
    def get_global_id(self, dim):
        """Equivalente à função get_global_id do OpenCL"""
        return get_global_id(dim)
    
    def get_local_id(self, dim):
        """Equivalente à função get_local_id do OpenCL"""
        return get_local_id(dim)
    
    def get_group_id(self, dim):
        """Equivalente à função get_group_id do OpenCL"""
        return get_group_id(dim)
    
    def get_local_size(self, dim):
        """Equivalente à função get_local_size do OpenCL"""
        return get_local_size(dim)
    
    def get_global_size(self, dim):
        """Equivalente à função get_global_size do OpenCL"""
        return get_global_size(dim)
    
    def barrier(self):
        """Equivalente à função barrier do OpenCL"""
        barrier()
    
    # Métodos de gestão de profiling e eventos
    def event(self):
        """Cria um novo evento"""
        if self.context is None:
            return None
        return Event(self.context)
    
    # Métodos de configuração
    def set_default_device(self, device_id):
        """Define o dispositivo padrão"""
        return self.select_device(device_id)
    
    def get_device_properties(self, device_id=None):
        """Retorna propriedades do dispositivo (compatível com CUDA)"""
        return self.get_device_info(device_id)
    
    def get_current_device_id(self):
        """Retorna o ID do dispositivo atual"""
        return self._current_device_id
    
    def print_device_info(self, device_id=None):
        """Imprime informações do dispositivo"""
        info = self.get_device_info(device_id)
        if not info:
            print("Dispositivo não disponível")
            return
            
        print("=== Informações do Dispositivo ===")
        print(f"ID: {info['id']}")
        print(f"Nome: {info['name']}")
        print(f"Tipo: {info['type']}")
        print(f"Plataforma: {info['platform']}")
        print(f"Fabricante: {info['vendor']}")
        print(f"Versão OpenCL: {info['version']}")
        print(f"Versão do Driver: {info['driver_version']}")
        print(f"Unidades de Computação: {info['compute_units']}")
        print(f"Tamanho Máximo de Grupo: {info['max_work_group_size']}")
        print(f"Tamanhos Máximos de Item: {info['max_work_item_sizes']}")
        print(f"Memória Global: {info['global_mem_size'] / (1024**2):.2f} MB")
        print(f"Memória Local: {info['local_mem_size'] / 1024:.2f} KB")
        print(f"Dispositivo Atual: {'Sim' if info['current'] else 'Não'}")


# Classe para representar arrays no dispositivo
class DeviceArray:
    def __init__(self, cl_buffer, shape, dtype, opencl_ext):
        self.cl_buffer = cl_buffer
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = dtype
        self.size = np.prod(shape)
        self.opencl_ext = opencl_ext
        self.ndim = len(self.shape)
        self._hash = None  # Para usar como chave de dicionário
    
    def copy_to_host(self):
        """Copia dados do dispositivo para o host"""
        if self.cl_buffer is None:
            print("Aviso: Buffer OpenCL é None, retornando array vazio")
            return np.empty(self.shape, dtype=self.dtype)
            
        try:
            result = np.empty(self.shape, dtype=self.dtype)
            cl.enqueue_copy(self.opencl_ext.queue, result, self.cl_buffer)
            self.opencl_ext.queue.finish()  # Garantir que a cópia seja concluída
            return result
        except cl.LogicError as e:
            print(f"Erro lógico do OpenCL ao copiar para o host: {e}")
            print("Isto pode ocorrer se o buffer foi invalidado ou o contexto foi perdido")
            return np.empty(self.shape, dtype=self.dtype)
        except cl.RuntimeError as e:
            print(f"Erro de runtime do OpenCL ao copiar para o host: {e}")
            return np.empty(self.shape, dtype=self.dtype)
        except Exception as e:
            print(f"Erro genérico ao copiar para o host: {e}")
            return np.empty(self.shape, dtype=self.dtype)
    
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
    
    def copy_to_device_async(self, arr, stream=None):
        """Copia dados do host para o dispositivo de forma assíncrona"""
        if self.cl_buffer is None:
            return self
            
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=self.dtype)
        
        if arr.shape != self.shape:
            raise ValueError(f"Forma incompatível: {arr.shape} vs {self.shape}")
        
        try:
            queue = stream.queue if stream else self.opencl_ext.queue
            cl.enqueue_copy(queue, self.cl_buffer, arr, is_blocking=False)
            return self
        except Exception as e:
            print(f"Erro ao copiar assincronamente para o dispositivo: {e}")
            return self
    
    def copy_from_device(self, device_array):
        """Copia dados de outro array no dispositivo"""
        if self.cl_buffer is None or device_array.cl_buffer is None:
            return self
            
        if device_array.shape != self.shape:
            raise ValueError(f"Forma incompatível: {device_array.shape} vs {self.shape}")
        
        try:
            cl.enqueue_copy(self.opencl_ext.queue, self.cl_buffer, device_array.cl_buffer)
            self.opencl_ext.queue.finish()
            return self
        except Exception as e:
            print(f"Erro ao copiar entre dispositivos: {e}")
            return self
    
    def copy_from_device_async(self, device_array, stream=None):
        """Copia dados de outro array no dispositivo de forma assíncrona"""
        if self.cl_buffer is None or device_array.cl_buffer is None:
            return self
            
        if device_array.shape != self.shape:
            raise ValueError(f"Forma incompatível: {device_array.shape} vs {self.shape}")
        
        try:
            queue = stream.queue if stream else self.opencl_ext.queue
            cl.enqueue_copy(queue, self.cl_buffer, device_array.cl_buffer, is_blocking=False)
            return self
        except Exception as e:
            print(f"Erro ao copiar assincronamente entre dispositivos: {e}")
            return self
    
    def __getitem__(self, key):
        """Suporte a indexação"""
        # Uma implementação completa gerenciaria a transferência parcial
        host_array = self.copy_to_host()
        return host_array[key]
    
    def __setitem__(self, key, value):
        """Suporte a atribuição por indexação"""
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
    
    def __hash__(self):
        """Suporte a uso como chave de dicionário"""
        if not self._hash:
            if SIPHASH_AVAILABLE:
                import siphash24
                # Usar siphash para hash confiável
                self._hash = siphash24.hash(str(id(self)))
            else:
                # Fallback para id
                self._hash = hash(id(self))
        return self._hash


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


# Classe para streams (equivalente a CommandQueue no OpenCL)
class Stream:
    def __init__(self, queue, stream_id, opencl_ext):
        self.queue = queue
        self.id = stream_id
        self.opencl_ext = opencl_ext
    
    def synchronize(self):
        """Sincroniza este stream"""
        self.queue.finish()
    
    def wait_event(self, event):
        """Aguarda um evento"""
        if event and event.event:
            event.event.wait()
    
    def __enter__(self):
        """Suporte a contexto 'with'"""
        self.old_stream = self.opencl_ext._active_stream
        self.opencl_ext._active_stream = self.queue
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restaura stream anterior ao sair do contexto"""
        self.opencl_ext._active_stream = self.old_stream


# Classe para eventos
class Event:
    def __init__(self, context):
        self.context = context
        self.event = None
        self._recorded = False
    
    def record(self, stream=None):
        """Registra um evento no stream"""
        if stream:
            queue = stream.queue
        else:
            # Usar queue padrão
            queue = opencl.queue
            
        # Criar um evento noop para marcar a posição atual
        self.event = cl.enqueue_marker(queue)
        self._recorded = True
        return self
    
    def synchronize(self):
        """Aguarda a conclusão do evento"""
        if self.event:
            self.event.wait()
    
    def elapsed_time(self, end_event):
        """Calcula o tempo decorrido entre este evento e outro"""
        if not self._recorded or not end_event._recorded:
            return 0.0
            
        if self.event and end_event.event:
            # Obter tempo em nanossegundos
            start = self.event.get_profiling_info(cl.profiling_info.END)
            end = end_event.event.get_profiling_info(cl.profiling_info.END)
            return (end - start) / 1e6  # nanossegundos para milissegundos
        return 0.0


# Funções auxiliares do OpenCL
def get_global_id(dim):
    """Emula a função get_global_id do OpenCL"""
    import pyopencl as cl
    try:
        return cl.get_global_id(dim)
    except:
        return 0  # Fallback

def get_local_id(dim):
    """Emula a função get_local_id do OpenCL"""
    import pyopencl as cl
    try:
        return cl.get_local_id(dim)
    except:
        return 0

def get_group_id(dim):
    """Emula a função get_group_id do OpenCL"""
    import pyopencl as cl
    try:
        return cl.get_group_id(dim)
    except:
        return 0

def get_local_size(dim):
    """Emula a função get_local_size do OpenCL"""
    import pyopencl as cl
    try:
        return cl.get_local_size(dim)
    except:
        return 1

def get_global_size(dim):
    """Emula a função get_global_size do OpenCL"""
    import pyopencl as cl
    try:
        return cl.get_global_size(dim)
    except:
        return 1

def barrier():
    """Emula a função barrier do OpenCL"""
    import pyopencl as cl
    try:
        return cl.barrier(cl.mem_flags.CLK_GLOBAL_MEM_FENCE)
    except:
        pass  # Fallback silencioso

# Criar instância global do módulo opencl
opencl = OpenCLExtension()
