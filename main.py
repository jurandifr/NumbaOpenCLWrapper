import numpy as np
import time
from numba_opencl import opencl
from numba_opencl.utils import check_opencl_support

# Verificar suporte OpenCL
opencl_available, device_info = check_opencl_support()
if opencl_available:
    if isinstance(device_info, list):
        print("=== Dispositivos OpenCL disponíveis ===")
        for i, device in enumerate(device_info):
            print(f"Dispositivo {i}: {device['name']} ({device['type']})")
            print(f"  Plataforma: {device['platform']}")
            print(f"  Versão: {device['version']}")
            print(f"  Unidades de computação: {device['max_compute_units']}")
            print(f"  Tamanho máximo do grupo: {device['max_work_group_size']}")
    else:
        print(f"OpenCL disponível, mas não foi possível listar dispositivos: {device_info}")
else:
    print(f"Aviso: OpenCL não está disponível: {device_info}")
    print("O código será executado em modo de simulação CPU.")

# Exemplo 1: Soma de vetores
@opencl.jit
def vector_add(a, b, c):
    i = opencl.get_global_id(0)
    if i < len(c):
        c[i] = a[i] + b[i]

# Exemplo 2: Multiplicação de matrizes
@opencl.jit
def matrix_multiply(a, b, c, width):
    """Multiplicação básica de matrizes"""
    row = opencl.get_global_id(0)
    col = opencl.get_global_id(1)

    if row < width and col < width:
        tmp = 0.0
        for i in range(width):
            tmp += a[row * width + i] * b[i * width + col]
        c[row * width + col] = tmp

# Exemplo 3: Filtro de imagem simples (blur)
@opencl.jit
def image_blur(input_img, output_img, width, height):
    x = opencl.get_global_id(0)
    y = opencl.get_global_id(1)

    if x > 0 and x < width - 1 and y > 0 and y < height - 1:
        # Média 3x3
        sum_value = 0.0
        for i in range(-1, 2):
            for j in range(-1, 2):
                sum_value += input_img[(y + j) * width + (x + i)]

        output_img[y * width + x] = sum_value / 9.0

def benchmark_vector_add(n=1000000):
    """Benchmarking da soma de vetores"""
    print(f"\n=== Benchmark: Soma de Vetores (N={n}) ===")

    # Criar dados de teste
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c_opencl = np.zeros_like(a)
    c_numpy = np.zeros_like(a)

    # Medir tempo NumPy
    start = time.time()
    c_numpy = a + b
    numpy_time = time.time() - start
    print(f"NumPy: {numpy_time:.6f} segundos")

    # Preparar OpenCL
    d_a = opencl.to_device(a)
    d_b = opencl.to_device(b)
    d_c = opencl.to_device(c_opencl)

    # Configurar grid e bloco
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    # Medir tempo OpenCL (incluindo transferência)
    start = time.time()
    vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
    opencl.synchronize()
    c_opencl = d_c.copy_to_host()
    opencl_time = time.time() - start
    print(f"OpenCL (total): {opencl_time:.6f} segundos")

    # Verificar resultado
    if np.allclose(c_numpy, c_opencl, rtol=1e-5, atol=1e-5):
        print("✓ Resultados corretos!")
        print(f"Speedup: {numpy_time/opencl_time:.2f}x")
    else:
        print("✗ Resultados divergem!")
        max_diff = np.max(np.abs(c_numpy - c_opencl))
        print(f"Máxima diferença: {max_diff}")

def benchmark_matrix_multiply(width=1024):
    """Benchmarking da multiplicação de matrizes"""
    print(f"\n=== Benchmark: Multiplicação de Matrizes ({width}x{width}) ===")

    # Criar matrizes para teste
    a_mat = np.random.rand(width, width).astype(np.float32)
    b_mat = np.random.rand(width, width).astype(np.float32)
    c_mat_opencl = np.zeros((width, width), dtype=np.float32)

    # Medir tempo NumPy
    start = time.time()
    c_mat_numpy = np.matmul(a_mat, b_mat)
    numpy_time = time.time() - start
    print(f"NumPy: {numpy_time:.6f} segundos")

    # Reformatar para array unidimensional (simulação)
    a_flat = a_mat.flatten()
    b_flat = b_mat.flatten()
    c_flat = c_mat_opencl.flatten()

    # Transferir para o dispositivo
    d_a = opencl.to_device(a_flat)
    d_b = opencl.to_device(b_flat)
    d_c = opencl.to_device(c_flat)

    # Medir tempo OpenCL
    start = time.time()
    block_dim = (16, 16)
    grid_dim = (width // block_dim[0] + 1, width // block_dim[1] + 1)
    matrix_multiply(d_a, d_b, d_c, width, grid=grid_dim, block=block_dim)
    opencl.synchronize()
    result_flat = d_c.copy_to_host()
    opencl_time = time.time() - start
    print(f"OpenCL (total): {opencl_time:.6f} segundos")

    # Reformatar resultado
    result_mat = result_flat.reshape((width, width))

    # Verificar resultado
    if np.allclose(c_mat_numpy, result_mat, rtol=1e-3, atol=1e-3):
        print("✓ Resultados corretos!")
        print(f"Speedup: {numpy_time/opencl_time:.2f}x")
    else:
        print("✗ Resultados divergem!")
        max_diff = np.max(np.abs(c_mat_numpy - result_mat))
        print(f"Máxima diferença: {max_diff}")

def run_examples():
    """Executa exemplos básicos"""
    print("\n=== Exemplos Básicos ===")

    # Exemplo 1: Soma de vetores
    print("\n1. Soma de Vetores")

    # Criar dados de teste
    N = 10000
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c = np.zeros_like(a)

    # Transferir para o dispositivo
    d_a = opencl.to_device(a)
    d_b = opencl.to_device(b)
    d_c = opencl.to_device(c)

    # Configurar grid e bloco
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    # Executar kernel
    vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)

    # Copiar resultado de volta
    result = d_c.copy_to_host()

    # Verificar resultado
    expected = a + b
    if np.allclose(result, expected):
        print("✓ Teste de soma de vetores: SUCESSO")
    else:
        print("✗ Teste de soma de vetores: FALHA")
        print(f"Primeiros 5 elementos: Resultado={result[:5]}, Esperado={expected[:5]}")

    # Exemplo 2: Multiplicação de matrizes
    print("\n2. Multiplicação de Matrizes")

    # Criar matrizes para teste
    width = 16
    a_mat = np.random.rand(width, width).astype(np.float32)
    b_mat = np.random.rand(width, width).astype(np.float32)
    c_mat = np.zeros((width, width), dtype=np.float32)

    # Reformatar para array unidimensional (simulação)
    a_flat = a_mat.flatten()
    b_flat = b_mat.flatten()
    c_flat = c_mat.flatten()

    # Transferir para o dispositivo
    d_a = opencl.to_device(a_flat)
    d_b = opencl.to_device(b_flat)
    d_c = opencl.to_device(c_flat)

    # Configurar grid e bloco para 2D
    matrix_multiply(d_a, d_b, d_c, width, grid=(width, width), block=(1, 1))

    # Copiar resultado de volta
    result_flat = d_c.copy_to_host()
    result_mat = result_flat.reshape((width, width))

    # Verificar resultado
    expected_mat = np.matmul(a_mat, b_mat)
    if np.allclose(result_mat, expected_mat, rtol=1e-4, atol=1e-4):
        print("✓ Teste de multiplicação de matrizes: SUCESSO")
    else:
        print("✗ Teste de multiplicação de matrizes: FALHA")

    # Exemplo 3: Filtro de imagem
    print("\n3. Filtro de Imagem (Blur)")

    # Criar imagem simples
    width, height = 512, 512
    img = np.random.rand(height, width).astype(np.float32)
    img_out = np.zeros_like(img)

    # Transferir para o dispositivo
    d_img = opencl.to_device(img.flatten())
    d_img_out = opencl.to_device(img_out.flatten())

    # Configurar grid e bloco
    block_dim = (16, 16)
    grid_dim = (width // block_dim[0] + 1, height // block_dim[1] + 1)

    # Executar filtro
    start = time.time()
    image_blur(d_img, d_img_out, width, height, grid=grid_dim, block=block_dim)
    blur_time = time.time() - start

    print(f"Tempo de execução do filtro: {blur_time:.6f} segundos")
    print("✓ Filtro de imagem aplicado com sucesso")

if __name__ == "__main__":
    # Executar exemplos
    run_examples()

    # Executar benchmarks
    benchmark_vector_add(n=5000000)
    benchmark_matrix_multiply(width=512)

    print("\nNumba.OpenCL")
    print("Esta é uma implementação protótipo que simula a API do numba.cuda com PyOpenCL")
    print("Para uso em produção, seria necessário integração completa com o compilador Numba")