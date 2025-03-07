
"""
Ferramentas de profiling para numba_opencl.

Este módulo fornece funcionalidades para analisar o desempenho
de kernels OpenCL, incluindo medição de tempo e detecção de gargalos.
"""

import time
import numpy as np
import threading
from collections import defaultdict
from prettytable import PrettyTable

class KernelProfiler:
    """
    Profiler para kernels OpenCL.
    
    Rastreia o tempo de execução de kernels e fornece estatísticas.
    """
    
    def __init__(self):
        self._stats = defaultdict(list)
        self._active = False
        self._lock = threading.Lock()
    
    def start(self):
        """Inicia o profiling."""
        self._active = True
        self._stats.clear()
    
    def stop(self):
        """Para o profiling."""
        self._active = False
    
    def clear(self):
        """Limpa as estatísticas coletadas."""
        with self._lock:
            self._stats.clear()
    
    def is_active(self):
        """Verifica se o profiling está ativo."""
        return self._active
    
    def record_kernel_execution(self, kernel_name, execution_time, grid_size, block_size, args_info=None):
        """
        Registra a execução de um kernel.
        
        Args:
            kernel_name: Nome do kernel
            execution_time: Tempo de execução em segundos
            grid_size: Tamanho da grade
            block_size: Tamanho do bloco
            args_info: Informações sobre os argumentos (opcional)
        """
        if not self._active:
            return
            
        with self._lock:
            self._stats[kernel_name].append({
                'time': execution_time,
                'grid': grid_size,
                'block': block_size,
                'args': args_info,
                'timestamp': time.time()
            })
    
    def get_stats(self, kernel_name=None):
        """
        Obtém estatísticas de execução.
        
        Args:
            kernel_name: Nome do kernel (opcional, se None retorna para todos)
            
        Returns:
            Dicionário com estatísticas
        """
        with self._lock:
            if kernel_name:
                if kernel_name in self._stats:
                    kernel_data = self._stats[kernel_name]
                    times = [run['time'] for run in kernel_data]
                    return {
                        'count': len(times),
                        'avg_time': np.mean(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'total_time': np.sum(times),
                        'std_dev': np.std(times),
                        'last_grid': kernel_data[-1]['grid'] if kernel_data else None,
                        'last_block': kernel_data[-1]['block'] if kernel_data else None
                    }
                return None
            
            # Estatísticas para todos os kernels
            result = {}
            for name, kernel_data in self._stats.items():
                times = [run['time'] for run in kernel_data]
                result[name] = {
                    'count': len(times),
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times),
                    'std_dev': np.std(times),
                    'last_grid': kernel_data[-1]['grid'] if kernel_data else None,
                    'last_block': kernel_data[-1]['block'] if kernel_data else None
                }
            return result
    
    def print_stats(self, kernel_name=None):
        """
        Imprime estatísticas formatadas.
        
        Args:
            kernel_name: Nome do kernel (opcional)
        """
        stats = self.get_stats(kernel_name)
        
        if kernel_name and not stats:
            print(f"Nenhuma estatística disponível para o kernel '{kernel_name}'")
            return
            
        if not stats:
            print("Nenhuma estatística de kernel disponível")
            return
            
        if kernel_name:
            print(f"Estatísticas para o kernel '{kernel_name}':")
            print(f"  Execuções: {stats['count']}")
            print(f"  Tempo médio: {stats['avg_time']*1000:.3f} ms")
            print(f"  Tempo mínimo: {stats['min_time']*1000:.3f} ms")
            print(f"  Tempo máximo: {stats['max_time']*1000:.3f} ms")
            print(f"  Tempo total: {stats['total_time']*1000:.3f} ms")
            print(f"  Desvio padrão: {stats['std_dev']*1000:.3f} ms")
            print(f"  Último grid: {stats['last_grid']}")
            print(f"  Último bloco: {stats['last_block']}")
        else:
            # Tabela para todos os kernels
            table = PrettyTable()
            table.field_names = [
                "Kernel", "Execuções", "Tempo Médio (ms)", 
                "Min (ms)", "Max (ms)", "Total (ms)", "% Total"
            ]
            
            # Calcular tempo total para percentagem
            total_all_kernels = sum(k['total_time'] for k in stats.values())
            
            # Ordenar por tempo total (decrescente)
            sorted_stats = sorted(
                [(name, data) for name, data in stats.items()],
                key=lambda x: x[1]['total_time'],
                reverse=True
            )
            
            for name, data in sorted_stats:
                percentage = (data['total_time'] / total_all_kernels * 100) if total_all_kernels > 0 else 0
                table.add_row([
                    name,
                    data['count'],
                    f"{data['avg_time']*1000:.3f}",
                    f"{data['min_time']*1000:.3f}",
                    f"{data['max_time']*1000:.3f}",
                    f"{data['total_time']*1000:.3f}",
                    f"{percentage:.1f}%"
                ])
            
            print("Estatísticas de kernels OpenCL:")
            print(table)
            print(f"Tempo total de todos os kernels: {total_all_kernels*1000:.3f} ms")

# Criar instância global do profiler
profiler = KernelProfiler()
