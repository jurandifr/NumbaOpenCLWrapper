
"""
Testes automatizados para o módulo numba_opencl.

Executa uma série de testes para verificar a funcionalidade básica,
compatibilidade com API CUDA e performance.
"""

import unittest
import numpy as np
import time
import sys
import os
from numba_opencl import ocl
from numba_opencl.utils import check_opencl_support, compare_arr

class TestOpenCLSupport(unittest.TestCase):
    """Testes de verificação de suporte OpenCL."""
    
    def test_opencl_available(self):
        """Verifica se o OpenCL está disponível no sistema."""
        available, info = check_opencl_support()
        print(f"OpenCL disponível: {available}")
        if available:
            if isinstance(info, list):
                print(f"Dispositivos encontrados: {len(info)}")
                for device in info:
                    print(f"  - {device['name']} ({device['type']}) - {device['platform']}")
            else:
                print(f"Informações: {info}")
        else:
            print(f"Motivo: {info}")
    
    def test_current_device(self):
        """Verifica o dispositivo atual."""
        device_id = opencl.get_current_device_id()
        print(f"ID do dispositivo atual: {device_id}")
        if device_id >= 0:
            info = opencl.get_device_info()
            print(f"Dispositivo: {info['name']}")
            print(f"Tipo: {info['type']}")
            print(f"Plataforma: {info['platform']}")
            print(f"Unidades de computação: {info['compute_units']}")

class TestBasicOperations(unittest.TestCase):
    """Testes de operações básicas."""
    
    def setUp(self):
        """Inicializa dados de teste."""
        self.test_size = 1000
        self.a = np.random.rand(self.test_size).astype(np.float32)
        self.b = np.random.rand(self.test_size).astype(np.float32)
        self.c = np.zeros(self.test_size, dtype=np.float32)
    
    def test_to_device(self):
        """Testa transferência para o dispositivo."""
        d_a = opencl.to_device(self.a)
        self.assertEqual(d_a.shape, self.a.shape)
        self.assertEqual(d_a.dtype, self.a.dtype)
        
        # Testar transferência de volta
        a_copy = d_a.copy_to_host()
        self.assertTrue(np.array_equal(self.a, a_copy))
    
    def test_vector_add(self):
        """Testa kernel de soma de vetores."""
        @opencl.jit
        def vector_add(a, b, c):
            i = opencl.get_global_id(0)
            if i < len(c):
                c[i] = a[i] + b[i]
        
        # Transferir para dispositivo
        d_a = opencl.to_device(self.a)
        d_b = opencl.to_device(self.b)
        d_c = opencl.to_device(self.c)
        
        # Executar kernel
        block_size = 256
        grid_size = (self.test_size + block_size - 1) // block_size
        vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        
        # Verificar resultado
        result = d_c.copy_to_host()
        expected = self.a + self.b
        
        self.assertTrue(np.allclose(result, expected, rtol=1e-5, atol=1e-5))
    
    def test_device_array(self):
        """Testa alocação de array no dispositivo."""
        d_array = opencl.device_array(self.test_size, dtype=np.float32)
        self.assertEqual(d_array.shape, (self.test_size,))
        self.assertEqual(d_array.dtype, np.float32)
        
        # Testar preenchimento com zeros
        array_data = d_array.copy_to_host()
        self.assertEqual(array_data.shape, (self.test_size,))

class TestMemoryOperations(unittest.TestCase):
    """Testes de operações de memória."""
    
    def setUp(self):
        """Inicializa dados de teste."""
        self.test_size = 1000
        self.data = np.random.rand(self.test_size).astype(np.float32)
    
    def test_copy_operations(self):
        """Testa operações de cópia."""
        # Transferir para dispositivo
        d_data = opencl.to_device(self.data)
        
        # Criar outro array no dispositivo
        d_copy = opencl.device_array_like(self.data)
        
        # Copiar entre dispositivos
        d_copy.copy_from_device(d_data)
        
        # Verificar resultado
        result = d_copy.copy_to_host()
        self.assertTrue(np.array_equal(result, self.data))
    
    def test_async_copy(self):
        """Testa operações de cópia assíncrona."""
        # Criar stream
        stream = opencl.stream()
        
        # Transferir para dispositivo de forma assíncrona
        d_data = opencl.to_device(self.data)
        
        # Criar outro array no dispositivo
        d_copy = opencl.device_array_like(self.data)
        
        # Copiar de forma assíncrona
        with stream:
            d_copy.copy_from_device_async(d_data, stream)
        
        # Sincronizar
        stream.synchronize()
        
        # Verificar resultado
        result = d_copy.copy_to_host()
        self.assertTrue(np.array_equal(result, self.data))

class TestAdvancedOperations(unittest.TestCase):
    """Testes de operações avançadas."""
    
    def test_matrix_multiply(self):
        """Testa multiplicação de matrizes."""
        width = 32
        a_mat = np.random.rand(width, width).astype(np.float32)
        b_mat = np.random.rand(width, width).astype(np.float32)
        c_mat = np.zeros((width, width), dtype=np.float32)
        
        @opencl.jit
        def matrix_multiply(a, b, c, width):
            row = opencl.get_global_id(0)
            col = opencl.get_global_id(1)
            
            if row < width and col < width:
                tmp = 0.0
                for i in range(width):
                    tmp += a[row * width + i] * b[i * width + col]
                c[row * width + col] = tmp
        
        # Reformatar para array unidimensional
        a_flat = a_mat.flatten()
        b_flat = b_mat.flatten()
        c_flat = c_mat.flatten()
        
        # Transferir para o dispositivo
        d_a = opencl.to_device(a_flat)
        d_b = opencl.to_device(b_flat)
        d_c = opencl.to_device(c_flat)
        
        # Executar kernel
        block_dim = (16, 16)
        grid_dim = (width // block_dim[0] + 1, width // block_dim[1] + 1)
        matrix_multiply(d_a, d_b, d_c, width, grid=grid_dim, block=block_dim)
        
        # Verificar resultado
        result_flat = d_c.copy_to_host()
        result_mat = result_flat.reshape((width, width))
        
        expected_mat = np.matmul(a_mat, b_mat)
        self.assertTrue(np.allclose(result_mat, expected_mat, rtol=1e-3, atol=1e-3))
    
    def test_reduction(self):
        """Testa redução (soma)."""
        data_size = 1024
        data = np.random.rand(data_size).astype(np.float32)
        
        @opencl.jit
        def reduction_sum(input_array, output_array, n):
            local_id = opencl.get_local_id(0)
            global_id = opencl.get_global_id(0)
            group_id = opencl.get_group_id(0)
            local_size = opencl.get_local_size(0)
            
            # Memória compartilhada para redução local
            shared = opencl.shared_array((local_size,), np.float32)
            
            # Carregar dados para memória compartilhada
            if global_id < n:
                shared[local_id] = input_array[global_id]
            else:
                shared[local_id] = 0.0
            
            # Sincronizar threads locais
            opencl.barrier()
            
            # Redução na memória compartilhada
            stride = local_size // 2
            while stride > 0:
                if local_id < stride:
                    shared[local_id] += shared[local_id + stride]
                opencl.barrier()
                stride //= 2
            
            # Primeiro thread de cada grupo escreve resultado parcial
            if local_id == 0:
                output_array[group_id] = shared[0]
        
        # Transferir para o dispositivo
        d_data = opencl.to_device(data)
        
        # Configurar tamanhos
        block_size = 256
        grid_size = (data_size + block_size - 1) // block_size
        
        # Alocar array para resultados parciais
        d_partial_sums = opencl.device_array((grid_size,), dtype=np.float32)
        
        # Executar redução
        reduction_sum(d_data, d_partial_sums, data_size, grid=grid_size, block=block_size)
        
        # Obter resultados parciais
        partial_sums = d_partial_sums.copy_to_host()
        
        # Somar resultados parciais
        sum_opencl = np.sum(partial_sums)
        sum_numpy = np.sum(data)
        
        self.assertTrue(np.isclose(sum_opencl, sum_numpy, rtol=1e-5))

class TestPerformance(unittest.TestCase):
    """Testes de performance."""
    
    def test_vector_add_performance(self):
        """Compara performance de soma de vetores entre OpenCL e NumPy."""
        n = 1000000
        a = np.random.rand(n).astype(np.float32)
        b = np.random.rand(n).astype(np.float32)
        
        # Medir tempo NumPy
        start = time.time()
        c_numpy = a + b
        numpy_time = time.time() - start
        
        @opencl.jit
        def vector_add(a, b, c):
            i = opencl.get_global_id(0)
            if i < len(c):
                c[i] = a[i] + b[i]
        
        # Transferir para dispositivo
        d_a = opencl.to_device(a)
        d_b = opencl.to_device(b)
        d_c = opencl.device_array_like(a)
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        
        # Aquecer GPU (primeira execução é mais lenta)
        vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        opencl.synchronize()
        
        # Medir tempo OpenCL
        start = time.time()
        vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        opencl.synchronize()
        opencl_time = time.time() - start
        
        print(f"\nTempo NumPy: {numpy_time:.6f} segundos")
        print(f"Tempo OpenCL: {opencl_time:.6f} segundos")
        if opencl_time < numpy_time:
            speedup = numpy_time / opencl_time
            print(f"OpenCL é {speedup:.2f}x mais rápido")
        else:
            slowdown = opencl_time / numpy_time
            print(f"OpenCL é {slowdown:.2f}x mais lento")

def run_tests():
    """Executa todos os testes."""
    test_classes = [
        TestOpenCLSupport,
        TestBasicOperations,
        TestMemoryOperations,
        TestAdvancedOperations,
        TestPerformance
    ]
    
    loader = unittest.TestLoader()
    suite_list = []
    
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suite_list.append(suite)
    
    full_suite = unittest.TestSuite(suite_list)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(full_suite)

if __name__ == "__main__":
    print("=" * 70)
    print("Testes do módulo numba_opencl")
    print("=" * 70)
    
    # Verificar disponibilidade do OpenCL
    available, info = check_opencl_support()
    if not available:
        print(f"Aviso: OpenCL não está disponível: {info}")
        print("Os testes serão executados em modo de simulação CPU.")
    
    run_tests()
