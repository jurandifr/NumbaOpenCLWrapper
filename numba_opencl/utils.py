"""
Utilitários para a extensão numba_opencl.

Este módulo fornece funções auxiliares usadas pelo módulo principal
para inspeção de código-fonte, geração de identificadores aleatórios
e outras funcionalidades de suporte.
"""

import inspect
import random
import string
import os
import sys
import numpy as np
import importlib

def inspect_function_source(func):
    """
    Extrai o código-fonte de uma função Python.

    Args:
        func: Função Python a ser inspecionada

    Returns:
        str: Código-fonte da função como string
    """
    return inspect.getsource(func)

def random_id(length=8):
    """
    Gera um ID aleatório com o comprimento especificado.

    Args:
        length (int): Comprimento do ID a ser gerado

    Returns:
        str: ID aleatório composto de letras e números
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def check_opencl_support():
    """
    Verifica se o OpenCL está disponível no sistema.

    Returns:
        tuple: (disponibilidade, informações)
            - disponibilidade (bool): True se o OpenCL estiver disponível, False caso contrário
            - informações: string com o motivo se indisponível, ou lista de dispositivos se disponível
    """
    try:
        import pyopencl as cl

        # Verificar se há plataformas OpenCL
        platforms = cl.get_platforms()
        if not platforms:
            return False, "Nenhuma plataforma OpenCL encontrada"

        # Coletar informações dos dispositivos
        devices_info = []

        for platform in platforms:
            # Tentar obter dispositivos GPU, CPU e aceleradores
            for device_type in [cl.device_type.GPU, cl.device_type.CPU, cl.device_type.ACCELERATOR]:
                try:
                    devices = platform.get_devices(device_type=device_type)
                    for device in devices:
                        type_str = cl.device_type.to_string(device.type)
                        # Coletar informações com tratamento de exceções para cada atributo
                        device_info = {'name': 'Desconhecido', 'type': type_str, 'platform': platform.name}

                        # Coletar atributos com tratamento de erro individual
                        try:
                            device_info['name'] = device.name
                        except:
                            device_info['name'] = f"Dispositivo {device_type} (nome indisponível)"

                        try:
                            device_info['version'] = device.version
                        except:
                            device_info['version'] = "Versão indisponível"

                        try:
                            device_info['max_compute_units'] = device.max_compute_units
                        except:
                            device_info['max_compute_units'] = 0

                        try:
                            device_info['max_work_group_size'] = device.max_work_group_size
                        except:
                            device_info['max_work_group_size'] = 256  # valor padrão seguro

                        # Informações adicionais de capacidade
                        try:
                            device_info['global_mem_size'] = device.global_mem_size
                        except:
                            device_info['global_mem_size'] = 0

                        try:
                            device_info['local_mem_size'] = device.local_mem_size
                        except:
                            device_info['local_mem_size'] = 0

                        try:
                            device_info['max_work_item_dimensions'] = device.max_work_item_dimensions
                        except:
                            device_info['max_work_item_dimensions'] = 3  # valor padrão

                        try:
                            device_info['vendor'] = device.vendor
                        except:
                            device_info['vendor'] = "Desconhecido"

                        devices_info.append(device_info)
                except cl.LogicError as e:
                    # Registrar mas continuar tentando outros tipos de dispositivo
                    print(f"Aviso: Erro ao consultar dispositivos {cl.device_type.to_string(device_type)} na plataforma {platform.name}: {e}")
                except Exception as e:
                    # Erro genérico ao acessar tipo de dispositivo específico
                    print(f"Aviso: Exceção ao consultar dispositivos {cl.device_type.to_string(device_type)}: {e}")

        if not devices_info:
            return False, "Nenhum dispositivo OpenCL encontrado"

        return True, devices_info

    except ImportError:
        return False, "Módulo pyopencl não está instalado"
    except cl.LogicError as e:
        return False, f"Erro lógico do OpenCL: {e}"
    except cl.RuntimeError as e:
        return False, f"Erro de execução do OpenCL: {e}"
    except Exception as e:
        return False, f"Erro ao verificar suporte OpenCL: {e}"

def get_env_device_selection():
    """
    Obtém a seleção de dispositivo a partir de variáveis de ambiente.

    Verifica se as variáveis OPENCL_DEVICE_TYPE ou OPENCL_DEVICE_ID estão definidas
    e retorna o valor apropriado.

    Returns:
        dict: Informações sobre a seleção de dispositivo das variáveis de ambiente
    """
    result = {'type': None, 'id': None}

    # Verificar variável OPENCL_DEVICE_TYPE
    device_type = os.environ.get('OPENCL_DEVICE_TYPE')
    if device_type:
        result['type'] = device_type.upper()

    # Verificar variável OPENCL_DEVICE_ID
    device_id = os.environ.get('OPENCL_DEVICE_ID')
    if device_id and device_id.isdigit():
        result['id'] = int(device_id)

    return result

def set_default_device_from_env(opencl_ext):
    """
    Configura o dispositivo padrão a partir de variáveis de ambiente.

    Args:
        opencl_ext: Instância de OpenCLExtension

    Returns:
        bool: True se um dispositivo foi configurado a partir de variáveis de ambiente
    """
    env_selection = get_env_device_selection()

    # Tentar selecionar pelo tipo de dispositivo
    if env_selection['type']:
        if opencl_ext.select_device_by_type(env_selection['type']):
            print(f"Dispositivo definido a partir da variável OPENCL_DEVICE_TYPE: {env_selection['type']}")
            return True

    # Tentar selecionar pelo ID do dispositivo
    if env_selection['id'] is not None:
        if opencl_ext.select_device(env_selection['id']):
            print(f"Dispositivo definido a partir da variável OPENCL_DEVICE_ID: {env_selection['id']}")
            return True

    return False

def compare_arrays(a, b, rtol=1e-5, atol=1e-8):
    """
    Compara dois arrays NumPy com tolerância.

    Args:
        a: Primeiro array
        b: Segundo array
        rtol: Tolerância relativa
        atol: Tolerância absoluta

    Returns:
        bool: True se os arrays são aproximadamente iguais
    """
    return np.allclose(a, b, rtol=rtol, atol=atol)

def format_size(size_bytes):
    """
    Formata um tamanho em bytes para uma string legível.

    Args:
        size_bytes: Tamanho em bytes

    Returns:
        str: Tamanho formatado (ex: "1.23 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"

def get_optimal_work_group_size(device, desired_size):
    """
    Calcula o tamanho de grupo de trabalho ótimo para um dispositivo.

    Args:
        device: Dispositivo OpenCL
        desired_size: Tamanho desejado

    Returns:
        int: Tamanho otimizado do grupo de trabalho
    """
    import pyopencl as cl

    try:
        max_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

        # Garantir que o tamanho esteja dentro dos limites do dispositivo
        if desired_size > max_size:
            return max_size

        # Para GPUs, geralmente queremos um múltiplo de 32 ou 64
        if device.get_info(cl.device_info.TYPE) == cl.device_type.GPU:
            # Encontrar o múltiplo de 32 mais próximo sem exceder o limite
            return min(((desired_size + 31) // 32) * 32, max_size)

        return desired_size
    except:
        # Fallback para um valor razoável
        return min(desired_size, 256)

def measure_performance(func, args=(), kwargs={}, warmup=1, repeat=3, number=1):
    """
    Mede o desempenho de uma função.
    
    Args:
        func: A função para medir o desempenho
        args: Tupla de argumentos para a função
        kwargs: Dicionário de argumentos nomeados para a função
        warmup: Número de execuções de aquecimento (não contabilizadas)
        repeat: Número de repetições de medição
        number: Número de chamadas por repetição
    
    Returns:
        dict: Resultados contendo tempos mínimo, máximo, médio e total
    """
    # Aquecimento
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Medição real
    times = []
    for _ in range(repeat):
        start = time.time()
        for _ in range(number):
            func(*args, **kwargs)
        end = time.time()
        times.append((end - start) / number)
    
    return {
        'min': min(times),
        'max': max(times),
        'avg': sum(times) / len(times),
        'total': sum(times),
        'times': times
    }
