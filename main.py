
import numpy as np
import time
from numba_opencl import opencl
from numba_opencl.utils import check_opencl_support

def print_device_selection_menu():
    """Exibe menu de seleção de dispositivos"""
    print("\n=== Seleção de Dispositivos OpenCL ===")
    
    devices = opencl.list_devices()
    if not devices:
        print("Nenhum dispositivo OpenCL detectado.")
        return False
    
    print("Dispositivos disponíveis:")
    for device in devices:
        current = " (atual)" if device['current'] else ""
        print(f"  {device['id']}: {device['name']} ({device['type']}) - {device['platform']}{current}")
    
    print("\nOpções:")
    print("  1. Selecionar dispositivo pelo ID")
    print("  2. Selecionar GPU (se disponível)")
    print("  3. Selecionar CPU (se disponível)")
    print("  4. Usar dispositivo atual")
    print("  5. Exibir informações detalhadas do dispositivo atual")
    
    try:
        choice = input("\nEscolha uma opção (1-5): ")
        
        if choice == "1":
            device_id = int(input("Digite o ID do dispositivo: "))
            if opencl.select_device(device_id):
                print(f"Dispositivo #{device_id} selecionado com sucesso.")
            else:
                print(f"Erro ao selecionar dispositivo #{device_id}.")
        elif choice == "2":
            if opencl.select_device_by_type('GPU'):
                print("Dispositivo GPU selecionado com sucesso.")
            else:
                print("Não foi possível selecionar um dispositivo GPU.")
        elif choice == "3":
            if opencl.select_device_by_type('CPU'):
                print("Dispositivo CPU selecionado com sucesso.")
            else:
                print("Não foi possível selecionar um dispositivo CPU.")
        elif choice == "4":
            print("Mantendo dispositivo atual.")
        elif choice == "5":
            opencl.print_device_info()
        else:
            print("Opção inválida.")
    except Exception as e:
        print(f"Erro na seleção: {e}")
    
    print("\nDispositivo atual:", end=" ")
    current_id = opencl.get_current_device_id()
    if current_id >= 0:
        info = opencl.get_device_info()
        print(f"{info['name']} ({info['type']}) - {info['platform']}")
    else:
        print("Nenhum dispositivo OpenCL (usando simulação CPU)")
    
    return True

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

# Exemplo 4: Redução (soma de todos os elementos)
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

def benchmark_reduction(n=1000000):
    """Benchmarking da redução (soma)"""
    print(f"\n=== Benchmark: Redução - Soma ({n} elementos) ===")

    # Criar dados de teste
    data = np.random.rand(n).astype(np.float32)

    # Resultado usando NumPy
    start = time.time()
    sum_numpy = np.sum(data)
    numpy_time = time.time() - start
    print(f"NumPy sum: {numpy_time:.6f} segundos, resultado: {sum_numpy}")

    # Transferir para o dispositivo
    d_data = opencl.to_device(data)

    # Configurar tamanhos
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    # Alocar array para resultados parciais
    d_partial_sums = opencl.device_array((grid_size,), dtype=np.float32)
    
    # Medir tempo OpenCL
    start = time.time()
    
    # Primeira redução
    reduction_sum(d_data, d_partial_sums, n, grid=grid_size, block=block_size)
    
    # Obter resultados parciais
    partial_sums = d_partial_sums.copy_to_host()
    
    # Somar resultados parciais na CPU
    sum_opencl = np.sum(partial_sums)
    opencl_time = time.time() - start
    
    print(f"OpenCL reduction: {opencl_time:.6f} segundos, resultado: {sum_opencl}")
    
    # Verificar resultado
    rel_error = abs(sum_numpy - sum_opencl) / abs(sum_numpy)
    if rel_error < 1e-5:
        print("✓ Resultados corretos!")
        print(f"Speedup: {numpy_time/opencl_time:.2f}x")
    else:
        print("✗ Resultados divergem!")
        print(f"Erro relativo: {rel_error}")

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
    
    # Exemplo 4: Redução
    print("\n4. Redução (Soma)")
    
    # Criar dados de teste
    data_size = 1024
    data = np.random.rand(data_size).astype(np.float32)
    
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
    
    # Somar resultados parciais na CPU
    sum_opencl = np.sum(partial_sums)
    sum_numpy = np.sum(data)
    
    print(f"Soma OpenCL: {sum_opencl}")
    print(f"Soma NumPy: {sum_numpy}")
    
    if abs(sum_opencl - sum_numpy) < 1e-5:
        print("✓ Teste de redução: SUCESSO")
    else:
        print("✗ Teste de redução: FALHA")

def test_streams():
    """Testa o uso de streams (execução concorrente)"""
    print("\n=== Teste de Streams (Execução Concorrente) ===")
    
    # Criar dados de teste
    N = 10000
    a1 = np.random.rand(N).astype(np.float32)
    b1 = np.random.rand(N).astype(np.float32)
    c1 = np.zeros_like(a1)
    
    a2 = np.random.rand(N).astype(np.float32)
    b2 = np.random.rand(N).astype(np.float32)
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
    grid_size = (N + block_size - 1) // block_size
    
    # Criar streams
    stream1 = opencl.stream()
    stream2 = opencl.stream()
    
    # Executar kernels em streams diferentes
    with stream1:
        vector_add(d_a1, d_b1, d_c1, grid=grid_size, block=block_size)
    
    with stream2:
        vector_add(d_a2, d_b2, d_c2, grid=grid_size, block=block_size)
    
    # Sincronizar
    stream1.synchronize()
    stream2.synchronize()
    
    # Copiar resultados
    result1 = d_c1.copy_to_host()
    result2 = d_c2.copy_to_host()
    
    # Verificar resultados
    expected1 = a1 + b1
    expected2 = a2 + b2
    
    if np.allclose(result1, expected1) and np.allclose(result2, expected2):
        print("✓ Teste de streams: SUCESSO")
    else:
        print("✗ Teste de streams: FALHA")

if __name__ == "__main__":
    # Mostrar menu para seleção de dispositivo
    print_device_selection_menu()
    
    # Executar exemplos
    run_examples()
    
    # Testar streams
    test_streams()

    # Executar benchmarks
    benchmark_vector_add(n=5000000)
    benchmark_matrix_multiply(width=512)
    benchmark_reduction(n=5000000)

    print("\nNumba.OpenCL")
    print("Esta é uma implementação protótipo que simula a API do numba.cuda com PyOpenCL")
    print("Para uso em produção, seria necessário integração completa com o compilador Numba")
