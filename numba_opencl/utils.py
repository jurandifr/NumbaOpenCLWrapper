
import string
import random
import inspect

def inspect_function_source(func):
    """Obtém o código fonte de uma função"""
    return inspect.getsource(func)

def random_id(length=8):
    """Gera um ID aleatório"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))
"""
Utilitários para a extensão numba_opencl.

Este módulo fornece funções auxiliares usadas pelo módulo principal
para inspeção de código-fonte, geração de identificadores aleatórios
e outras funcionalidades de suporte.
"""

import inspect
import random
import string

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
            - informações_dispositivos (list): Lista de dicionários com informações dos dispositivos
    """
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        
        if not platforms:
            return False, "Nenhuma plataforma OpenCL encontrada"
        
        devices_info = []
        for platform in platforms:
            platform_name = platform.get_info(cl.platform_info.NAME)
            devices = platform.get_devices()
            
            for device in devices:
                device_info = {
                    "platform": platform_name,
                    "name": device.get_info(cl.device_info.NAME),
                    "type": cl.device_type.to_string(device.get_info(cl.device_info.TYPE)),
                    "version": device.get_info(cl.device_info.VERSION),
                    "max_compute_units": device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                    "max_work_group_size": device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
                }
                devices_info.append(device_info)
        
        return True, devices_info
    except ImportError:
        return False, "PyOpenCL não está instalado"
    except Exception as e:
        return False, f"Erro ao verificar suporte OpenCL: {e}"
