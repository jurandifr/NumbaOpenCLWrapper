
import numpy as np
import time
from numba_opencl import opencl

def test_vector_add():
    """Testa a operação básica de soma de vetores"""
    print("\n=== Teste: Soma de Vetores ===")
    
    @opencl.jit
    def vector_add(a, b, c):
        i = opencl.get_global_id(0)
        if i < len(c):
            c[i] = a[i] + b[i]
    
    # Criar dados
    N = 1000000
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c = np.zeros_like(a)
    
    # Transferir para GPU
    d_a = opencl.to_device(a)
    d_b = opencl.to_device(b)
    d_c = opencl.to_device(c)
    
    # Executar kernel
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
    
    # Verificar resultado
    result = d_c.copy_to_host()
    expected = a + b
    
    if np.allclose(result, expected, rtol=1e-5, atol=1e-5):
        print("✅ Teste passou!")
    else:
        print("❌ Teste falhou!")
        print(f"Máxima diferença: {np.max(np.abs(result - expected))}")
    
    return np.allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_performance():
    """Testa o desempenho em comparação com NumPy"""
    print("\n=== Teste: Desempenho ===")
    
    @opencl.jit
    def vector_multiply(a, b, c):
        i = opencl.get_global_id(0)
        if i < len(c):
            c[i] = a[i] * b[i]
    
    # Criar dados grandes
    N = 10000000
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c_opencl = np.zeros_like(a)
    
    # Medir NumPy
    start = time.time()
    c_numpy = a * b
    numpy_time = time.time() - start
    print(f"NumPy: {numpy_time:.6f} segundos")
    
    # Medir OpenCL
    d_a = opencl.to_device(a)
    d_b = opencl.to_device(b)
    d_c = opencl.to_device(c_opencl)
    
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    
    start = time.time()
    vector_multiply(d_a, d_b, d_c, grid=grid_size, block=block_size)
    opencl.synchronize()
    result = d_c.copy_to_host()
    opencl_time = time.time() - start
    
    print(f"OpenCL: {opencl_time:.6f} segundos")
    print(f"Speedup: {numpy_time/opencl_time:.2f}x")
    
    return opencl_time < numpy_time  # Espera-se que OpenCL seja mais rápido

if __name__ == "__main__":
    print("Executando testes para numba_opencl...")
    
    # Executar testes
    tests_passed = []
    tests_passed.append(test_vector_add())
    tests_passed.append(test_performance())
    
    # Resumo
    print("\n=== Resumo dos Testes ===")
    print(f"Testes executados: {len(tests_passed)}")
    print(f"Testes bem-sucedidos: {sum(tests_passed)}")
    print(f"Testes falhos: {len(tests_passed) - sum(tests_passed)}")
    
    if all(tests_passed):
        print("\n✅ Todos os testes passaram!")
    else:
        print("\n❌ Alguns testes falharam!")
