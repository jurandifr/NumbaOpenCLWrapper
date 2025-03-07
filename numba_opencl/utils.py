
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
