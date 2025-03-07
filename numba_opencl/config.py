
"""
Módulo de configuração para numba_opencl.

Fornece funções para persistir e carregar configurações.
"""

import os
import json
import numpy as np
from pathlib import Path

# Caminho padrão para o arquivo de configuração
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.numba_opencl_config.json")

def get_default_config():
    """Retorna a configuração padrão."""
    return {
        "device_id": 0,
        "device_preference": ["GPU", "CPU", "ACCELERATOR"],
        "default_block_size": 256,
        "auto_select_device": True,
        "show_verbose_info": False,
        "profile_kernels": False,
        "use_fast_math": True
    }

def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    """
    Salva a configuração em um arquivo JSON.
    
    Args:
        config: Dicionário com a configuração
        config_path: Caminho para o arquivo de configuração
    """
    try:
        # Criar diretório se não existir
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Erro ao salvar configuração: {e}")
        return False

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Carrega a configuração de um arquivo JSON.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com a configuração, ou configuração padrão em caso de erro
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            return get_default_config()
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return get_default_config()

def benchmark_device(opencl_ext, device_id, test_size=1000000):
    """
    Realiza benchmark simples em um dispositivo.
    
    Args:
        opencl_ext: Instância do OpenCLExtension
        device_id: ID do dispositivo para testar
        test_size: Tamanho do array de teste
        
    Returns:
        Tempo de execução em segundos ou None em caso de erro
    """
    try:
        # Selecionar dispositivo
        if not opencl_ext.select_device(device_id):
            return None
            
        # Dados de teste
        a = np.random.rand(test_size).astype(np.float32)
        b = np.random.rand(test_size).astype(np.float32)
        
        # Transferir para dispositivo
        d_a = opencl_ext.to_device(a)
        d_b = opencl_ext.to_device(b)
        d_c = opencl_ext.device_array(test_size, dtype=np.float32)
        
        # Definir um kernel de teste simples
        @opencl_ext.jit
        def benchmark_kernel(a, b, c):
            i = opencl_ext.get_global_id(0)
            if i < len(c):
                c[i] = a[i] + b[i]
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (test_size + block_size - 1) // block_size
        
        # Medir tempo
        import time
        start = time.time()
        
        # Executar kernel
        benchmark_kernel(d_a, d_b, d_c, grid=grid_size, block=block_size)
        opencl_ext.synchronize()
        
        # Finalizar medição
        end = time.time()
        return end - start
        
    except Exception as e:
        print(f"Erro durante benchmark: {e}")
        return None

def auto_select_best_device(opencl_ext):
    """
    Seleciona automaticamente o melhor dispositivo disponível.
    
    Args:
        opencl_ext: Instância do OpenCLExtension
        
    Returns:
        ID do dispositivo selecionado, ou -1 se nenhum dispositivo disponível
    """
    if not opencl_ext.opencl_available or not opencl_ext._available_devices:
        return -1
        
    best_device = -1
    best_time = float('inf')
    
    print("Realizando benchmark para selecionar o melhor dispositivo...")
    
    # Testar cada dispositivo
    for i in range(opencl_ext.device_count()):
        device_info = opencl_ext.get_device_info(i)
        print(f"Testando {device_info['name']} ({device_info['type']})...")
        
        time_taken = benchmark_device(opencl_ext, i, test_size=100000)
        
        if time_taken is not None:
            print(f"Tempo: {time_taken:.6f} segundos")
            if time_taken < best_time:
                best_time = time_taken
                best_device = i
        else:
            print("Falha no teste")
    
    if best_device >= 0:
        device_info = opencl_ext.get_device_info(best_device)
        print(f"Melhor dispositivo: {device_info['name']} ({device_info['type']})")
        opencl_ext.select_device(best_device)
        
        # Atualizar configuração
        config = load_config()
        config['device_id'] = best_device
        save_config(config)
        
    return best_device
