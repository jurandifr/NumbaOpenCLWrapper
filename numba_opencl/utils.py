
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
    Verifica se o sistema suporta OpenCL e retorna informações sobre os dispositivos disponíveis.
    
    Returns:
        tuple: (suporte_disponível, informações_dispositivos)
            - suporte_disponível (bool): True se OpenCL está disponível
            - informações_dispositivos (list/str): Lista de dicionários com informações dos dispositivos ou mensagem de erro
    """
    # Verificar se PyOpenCL está instalado
    if not importlib.util.find_spec("pyopencl"):
        return False, "PyOpenCL não está instalado"
    
    try:
        import pyopencl as cl
        
        # Verificar se há plataformas OpenCL disponíveis
        platforms = cl.get_platforms()
        if not platforms:
            return False, "Nenhuma plataforma OpenCL encontrada no sistema"
        
        devices_info = []
        for platform in platforms:
            platform_name = platform.get_info(cl.platform_info.NAME)
            platform_vendor = platform.get_info(cl.platform_info.VENDOR)
            platform_version = platform.get_info(cl.platform_info.VERSION)
            
            # Tentar obter dispositivos de diferentes tipos
            for device_type_name, device_type in [
                ("GPU", cl.device_type.GPU),
                ("CPU", cl.device_type.CPU),
                ("ACCELERATOR", cl.device_type.ACCELERATOR)
            ]:
                try:
                    devices = platform.get_devices(device_type=device_type)
                    for device in devices:
                        device_info = {
                            "platform": platform_name,
                            "platform_vendor": platform_vendor,
                            "platform_version": platform_version,
                            "name": device.get_info(cl.device_info.NAME),
                            "type": device_type_name,
                            "vendor": device.get_info(cl.device_info.VENDOR),
                            "version": device.get_info(cl.device_info.VERSION),
                            "driver_version": device.get_info(cl.device_info.DRIVER_VERSION),
                            "max_compute_units": device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                            "max_work_group_size": device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE),
                            "max_work_item_sizes": device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES),
                            "global_mem_size": device.get_info(cl.device_info.GLOBAL_MEM_SIZE),
                            "local_mem_size": device.get_info(cl.device_info.LOCAL_MEM_SIZE),
                            "extensions": device.get_info(cl.device_info.EXTENSIONS).decode('utf-8').split(' ')
                        }
                        devices_info.append(device_info)
                except:
                    # Ignorar erros ao tentar obter dispositivos de um tipo específico
                    pass
        
        if not devices_info:
            return False, "Plataformas OpenCL encontradas, mas nenhum dispositivo compatível disponível"
        
        return True, devices_info
    
    except Exception as e:
        return False, f"Erro ao verificar suporte OpenCL: {str(e)}"

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
