
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
