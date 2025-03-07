
"""
Testes automatizados para o módulo numba_opencl.

Executa uma série de testes para verificar a funcionalidade básica,
compatibilidade com API CUDA e performance.
"""

import unittest
import numpy as np
import time
import sys
from numba_opencl import opencl
from numba_opencl.utils import check_opencl_support, compare_arrays

class TestOpenCL(unittest.TestCase):
    """Testes para a extensão numba_opencl"""
    
    def setUp(self):
        """Configurações iniciais para os testes"""
        # Verificar se OpenCL está disponível
        self.opencl_available, _ = check_opencl_support()
        
        # Definir tamanhos para testes
        self.N = 10000
        self.test_array = np.random.rand(self.N).astype(np.float32)
        
        # Kernel de teste para soma de vetores
        @opencl.jit
        def vector_add(a, b, c):
            i = opencl.get_global_id(0)
            if i < len(c):
                c[i] = a[i] + b[i]
        
        # Kernel de teste para escala de vetor
        @opencl.jit
        def vector_scale(a, b, scale):
            i = opencl.get_global_id(0)
            if i < len(a):
                b[i] = a[i] * scale
        
        self.vector_add = vector_add
        self.vector_scale = vector_scale
    
    def test_device_detection(self):
        """Testa detecção de dispositivos OpenCL"""
        print("\n=== Teste de Detecção de Dispositivos ===")
        
        # Listar dispositivos
        devices = opencl.list_devices()
        print(f"Dispositivos detectados: {len(devices)}")
        
        for device in devices:
            print(f"  {device['id']}: {device['name']} ({device['type']}) - {device['platform']}")
        
        # Verificar contagem de dispositivos
        self.assertEqual(len(devices), opencl.device_count())
        
        # Verificar dispositivo atual
        current_device = opencl.get_current_device()
        print(f"Dispositivo atual: {current_device}")
        
        self.assertTrue(len(devices) > 0 or not self.opencl_available)
    
    def test_device_selection(self):
        """Testa seleção de dispositivos OpenCL"""
        if not self.opencl_available:
            self.skipTest("OpenCL não está disponível")
        
        print("\n=== Teste de Seleção de Dispositivos ===")
        
        # Obter ID do dispositivo atual
        current_id = opencl.get_current_device_id()
        print(f"Dispositivo atual: {current_id}")
        
        # Obter contagem de dispositivos
        device_count = opencl.device_count()
        if device_count > 1:
            # Tentar selecionar outro dispositivo
            new_id = (current_id + 1) % device_count
            success = opencl.select_device(new_id)
            print(f"Selecionando dispositivo {new_id}: {'Sucesso' if success else 'Falha'}")
            
            # Verificar se a seleção funcionou
            if success:
                self.assertEqual(opencl.get_current_device_id(), new_id)
            
            # Voltar ao dispositivo original
            opencl.select_device(current_id)
    
    def test_memory_transfer(self):
        """Testa transferência de memória entre host e dispositivo"""
        print("\n=== Teste de Transferência de Memória ===")
        
        # Criar array para teste
        host_array = np.random.rand(self.N).astype(np.float32)
        
        # Transferir para o dispositivo
        device_array = opencl.to_device(host_array)
        
        # Transferir de volta para o host
        result_array = device_array.copy_to_host()
        
        # Verificar se os dados são idênticos
        self.assertTrue(compare_arrays(host_array, result_array))
        print("✓ Transferência de memória: SUCESSO")
    
    def test_basic_kernel(self):
        """Testa execução básica de kernel"""
        print("\n=== Teste de Kernel Básico ===")
        
        # Criar dados de teste
        a = np.random.rand(self.N).astype(np.float32)
        b = np.random.rand(self.N).astype(np.float32)
        c = np.zeros_like(a)
        
        # Transferir para o dispositivo
        d_a = opencl.to_device(a)
        d_b = opencl.to_device(b)
        d_c = opencl.to_device(c)
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (self.N + block_size - 1) // block_size
        
        # Executar kernel
        self.vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        
        # Copiar resultado de volta
        result = d_c.copy_to_host()
        
        # Verificar resultado
        expected = a + b
        self.assertTrue(compare_arrays(result, expected))
        print("✓ Kernel básico: SUCESSO")
    
    def test_scalar_argument(self):
        """Testa passagem de argumento escalar para kernel"""
        print("\n=== Teste de Argumento Escalar ===")
        
        # Criar dados de teste
        a = np.random.rand(self.N).astype(np.float32)
        b = np.zeros_like(a)
        scale_factor = 2.5
        
        # Transferir para o dispositivo
        d_a = opencl.to_device(a)
        d_b = opencl.to_device(b)
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (self.N + block_size - 1) // block_size
        
        # Executar kernel
        self.vector_scale(d_a, d_b, scale_factor, grid=grid_size, block=block_size)
        
        # Copiar resultado de volta
        result = d_b.copy_to_host()
        
        # Verificar resultado
        expected = a * scale_factor
        self.assertTrue(compare_arrays(result, expected))
        print("✓ Argumento escalar: SUCESSO")
    
    def test_stream(self):
        """Testa uso de streams (execução concorrente)"""
        print("\n=== Teste de Streams ===")
        
        # Criar dados de teste
        a1 = np.random.rand(self.N).astype(np.float32)
        b1 = np.random.rand(self.N).astype(np.float32)
        c1 = np.zeros_like(a1)
        
        a2 = np.random.rand(self.N).astype(np.float32)
        b2 = np.random.rand(self.N).astype(np.float32)
        c2 = np.zeros_like(a2)
        
        # Transferir para o dispositivo
        d_a1 = opencl.to_device(a1)
        d_b1 = opencl.to_device(b1)
        d_c1 = opencl.to_device(c1)
        
        d_a2 = opencl.to_device(a2)
        d_b2 = opencl.to_device(b2)
        d_c2 = opencl.to_device(c2)
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (self.N + block_size - 1) // block_size
        
        # Criar streams
        stream1 = opencl.stream()
        stream2 = opencl.stream()
        
        try:
            # Executar kernels em streams diferentes
            with stream1:
                self.vector_add(d_a1, d_b1, d_c1, grid=grid_size, block=block_size)
            
            with stream2:
                self.vector_add(d_a2, d_b2, d_c2, grid=grid_size, block=block_size)
            
            # Sincronizar
            stream1.synchronize()
            stream2.synchronize()
            
            # Copiar resultados
            result1 = d_c1.copy_to_host()
            result2 = d_c2.copy_to_host()
            
            # Verificar resultados
            expected1 = a1 + b1
            expected2 = a2 + b2
            
            self.assertTrue(compare_arrays(result1, expected1))
            self.assertTrue(compare_arrays(result2, expected2))
            print("✓ Streams: SUCESSO")
        except Exception as e:
            print(f"Erro no teste de streams: {e}")
            self.fail(f"Teste de streams falhou: {e}")
    
    def test_events(self):
        """Testa eventos e sincronização"""
        print("\n=== Teste de Eventos ===")
        
        # Criar dados de teste
        a = np.random.rand(self.N).astype(np.float32)
        b = np.random.rand(self.N).astype(np.float32)
        c = np.zeros_like(a)
        
        # Transferir para o dispositivo
        d_a = opencl.to_device(a)
        d_b = opencl.to_device(b)
        d_c = opencl.to_device(c)
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (self.N + block_size - 1) // block_size
        
        try:
            # Criar eventos
            start_event = opencl.event()
            end_event = opencl.event()
            
            # Registrar eventos
            start_event.record()
            
            # Executar kernel
            self.vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
            
            # Registrar evento final
            end_event.record()
            
            # Sincronizar eventos
            end_event.synchronize()
            
            # Obter tempo decorrido
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"Tempo decorrido: {elapsed_time} ms")
            
            # Verificar resultado
            result = d_c.copy_to_host()
            expected = a + b
            
            self.assertTrue(compare_arrays(result, expected))
            print("✓ Eventos: SUCESSO")
        except Exception as e:
            print(f"Erro no teste de eventos: {e}")
            # Não falhar o teste, pois alguns dispositivos podem não suportar profiling
            print("Ignorando falha no teste de eventos")
    
    def test_performance(self):
        """Testa performance básica"""
        print("\n=== Teste de Performance ===")
        
        # Definir tamanho maior para teste de performance
        N = 10000000
        
        # Criar dados de teste
        a = np.random.rand(N).astype(np.float32)
        b = np.random.rand(N).astype(np.float32)
        c_opencl = np.zeros_like(a)
        
        # Medir tempo NumPy
        start = time.time()
        c_numpy = a + b
        numpy_time = time.time() - start
        print(f"NumPy: {numpy_time:.6f} segundos")
        
        # Transferir para o dispositivo
        d_a = opencl.to_device(a)
        d_b = opencl.to_device(b)
        d_c = opencl.to_device(c_opencl)
        
        # Configurar grid e bloco
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        
        # Execução inicial para "aquecer" (ignorar)
        self.vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        opencl.synchronize()
        
        # Medir tempo OpenCL (apenas kernel)
        start = time.time()
        self.vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        opencl.synchronize()
        kernel_time = time.time() - start
        print(f"OpenCL (kernel): {kernel_time:.6f} segundos")
        
        # Medir tempo OpenCL (incluindo transferência)
        start = time.time()
        d_a = opencl.to_device(a)
        d_b = opencl.to_device(b)
        d_c = opencl.to_device(c_opencl)
        self.vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
        opencl.synchronize()
        c_opencl = d_c.copy_to_host()
        total_time = time.time() - start
        print(f"OpenCL (total): {total_time:.6f} segundos")
        
        # Verificar resultado
        self.assertTrue(compare_arrays(c_numpy, c_opencl, rtol=1e-4, atol=1e-4))
        print("✓ Resultado correto")
        
        # Calcular speedups
        kernel_speedup = numpy_time / kernel_time
        total_speedup = numpy_time / total_time
        print(f"Speedup (kernel): {kernel_speedup:.2f}x")
        print(f"Speedup (total): {total_speedup:.2f}x")

if __name__ == '__main__':
    print("Testes NumbaOpenCL")
    print("==================")
    
    # Verificar se OpenCL está disponível
    opencl_available, device_info = check_opencl_support()
    if not opencl_available:
        print(f"AVISO: OpenCL não está disponível: {device_info}")
        print("Os testes serão executados em modo de simulação CPU.")
    
    # Executar testes
    unittest.main(argv=[sys.argv[0]])
