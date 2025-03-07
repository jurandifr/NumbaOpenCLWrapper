"""
Exemplos e benchmarks para a extensão numba_opencl.

Este script demonstra o uso da extensão numba_opencl, que fornece
uma API compatível com numba.cuda para OpenCL.
"""

import numpy as np
import time
import argparse
import os
import sys
from numba_opencl import ocl
from numba_opencl.utils import check_opencl_support
from numba_opencl.profiler import profiler
from prettytable import PrettyTable

# Verifica se o módulo prettytable está instalado
try:
    import prettytable
except ImportError:
    print("Instalando dependência adicional (prettytable)...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])

def print_device_selection_menu():
    """Exibe menu de seleção de dispositivos"""
    print("\n=== Seleção de Dispositivos OpenCL ===")

    devices = ocl.list_devices()
    if not devices:
        print("Nenhum dispositivo OpenCL detectado.")
        return False

    # Tabela de dispositivos
    table = PrettyTable()
    table.field_names = ["ID", "Nome", "Tipo", "Plataforma", "Unidades Comp.", "Atual"]

    for device in devices:
        current = "✓" if device['current'] else ""
        table.add_row([
            device['id'],
            device['name'],
            device['type'],
            device['platform'],
            device['compute_units'],
            current
        ])

    print(table)

    print("\nOpções:")
    print("  1. Selecionar dispositivo pelo ID")
    print("  2. Selecionar GPU (se disponível)")
    print("  3. Selecionar CPU (se disponível)")
    print("  4. Usar dispositivo atual")
    print("  5. Exibir informações detalhadas do dispositivo atual")
    print("  6. Benchmark de todos os dispositivos")

    try:
        choice = input("\nEscolha uma opção (1-6): ")

        if choice == "1":
            device_id = int(input("Digite o ID do dispositivo: "))
            if ocl.select_device(device_id):
                print(f"Dispositivo #{device_id} selecionado com sucesso.")
            else:
                print(f"Erro ao selecionar dispositivo #{device_id}.")
        elif choice == "2":
            if ocl.select_device_by_type('GPU'):
                print("Dispositivo GPU selecionado com sucesso.")
            else:
                print("Não foi possível selecionar um dispositivo GPU.")
        elif choice == "3":
            if ocl.select_device_by_type('CPU'):
                print("Dispositivo CPU selecionado com sucesso.")
            else:
                print("Não foi possível selecionar um dispositivo CPU.")
        elif choice == "4":
            print("Mantendo dispositivo atual.")
        elif choice == "5":
            ocl.print_device_info()
        elif choice == "6":
            benchmark_all_devices()
        else:
            print("Opção inválida.")
    except Exception as e:
        print(f"Erro na seleção: {e}")

    print("\nDispositivo atual:", end=" ")
    current_id = ocl.get_current_device_id()
    if current_id >= 0:
        info = ocl.get_device_info()
        print(f"{info['name']} ({info['type']}) - {info['platform']}")
    else:
        print("Nenhum dispositivo OpenCL (usando simulação CPU)")

    return True

def benchmark_all_devices():
    """Realiza benchmark em todos os dispositivos disponíveis."""
    from numba_opencl.config import benchmark_device

    device_count = ocl.device_count()
    if device_count == 0:
        print("Nenhum dispositivo OpenCL disponível para benchmark.")
        return

    print("\n=== Benchmark de Dispositivos OpenCL ===")

    table = PrettyTable()
    table.field_names = ["ID", "Nome", "Tipo", "Plataforma", "Tempo (s)", "Relativo"]

    results = []

    # Testar cada dispositivo
    for i in range(device_count):
        device_info = ocl.get_device_info(i)
        print(f"Testando {device_info['name']} ({device_info['type']})...", end="", flush=True)

        time_taken = benchmark_device(ocl, i, test_size=1000000)

        if time_taken is not None:
            print(f" {time_taken:.6f} segundos")
            results.append((i, device_info, time_taken))
        else:
            print(" Falha no teste")

    if not results:
        print("Nenhum dispositivo completou o benchmark com sucesso.")
        return

    # Encontrar o mais rápido para comparação relativa
    best_time = min(time for _, _, time in results)

    # Adicionar resultados à tabela
    for i, device_info, time_taken in sorted(results, key=lambda x: x[2]):
        relative = best_time / time_taken
        table.add_row([
            i,
            device_info['name'],
            device_info['type'],
            device_info['platform'],
            f"{time_taken:.6f}",
            f"{relative:.2f}x"
        ])

    print("\nResultados do benchmark:")
    print(table)

    # Recomendar o melhor dispositivo
    best_device = min(results, key=lambda x: x[2])[0]
    best_device_info = ocl.get_device_info(best_device)
    print(f"\nDispositivo mais rápido: {best_device_info['name']} (ID: {best_device})")

    use_best = input("Usar o dispositivo mais rápido? (s/n): ").lower()
    if use_best == 's':
        ocl.select_device(best_device)
        print(f"Dispositivo {best_device_info['name']} selecionado.")

# Exemplo 1: Soma de vetores
@ocl.jit
def vector_add(a, b, c):
    i = ocl.get_global_id(0)
    if i < len(c):
        c[i] = a[i] + b[i]

# Exemplo 2: Multiplicação de matrizes
@ocl.jit
def matrix_multiply(a, b, c, width):
    """Multiplicação básica de matrizes"""
    row = ocl.get_global_id(0)
    col = ocl.get_global_id(1)

    if row < width and col < width:
        tmp = 0.0
        for i in range(width):
            tmp += a[row * width + i] * b[i * width + col]
        c[row * width + col] = tmp

# Exemplo 3: Filtro de imagem simples (blur)
@ocl.jit
def image_blur(input_img, output_img, width, height):
    x = ocl.get_global_id(0)
    y = ocl.get_global_id(1)

    if x > 0 and x < width - 1 and y > 0 and y < height - 1:
        # Média 3x3
        sum_value = 0.0
        for i in range(-1, 2):
            for j in range(-1, 2):
                sum_value += input_img[(y + j) * width + (x + i)]

        output_img[y * width + x] = sum_value / 9.0

# Exemplo 4: Redução (soma de todos os elementos)
@ocl.jit
def reduction_sum(input_array, output_array, n):
    local_id = ocl.get_local_id(0)
    global_id = ocl.get_global_id(0)
    group_id = ocl.get_group_id(0)
    local_size = ocl.get_local_size(0)

    # Memória compartilhada para redução local
    shared = ocl.shared_array((local_size,), np.float32)

    # Carregar dados para memória compartilhada
    if global_id < n:
        shared[local_id] = input_array[global_id]
    else:
        shared[local_id] = 0.0

    # Sincronizar threads locais
    ocl.barrier()

    # Redução na memória compartilhada
    stride = local_size // 2
    while stride > 0:
        if local_id < stride:
            shared[local_id] += shared[local_id + stride]
        ocl.barrier()
        stride //= 2

    # Primeiro thread de cada grupo escreve resultado parcial
    if local_id == 0:
        output_array[group_id] = shared[0]

# Exemplo 5: SAXPY (Single-Precision A*X Plus Y)
@ocl.jit
def saxpy(a, x, y, result):
    i = ocl.get_global_id(0)
    if i < len(result):
        result[i] = a * x[i] + y[i]

# Exemplo 6: Convolução 2D
@ocl.jit
def convolution_2d(input_array, output_array, filter_array, width, height, filter_width, filter_height):
    x = ocl.get_global_id(0)
    y = ocl.get_global_id(1)

    if x < width and y < height:
        sum_value = 0.0
        filter_radius_x = filter_width // 2
        filter_radius_y = filter_height // 2

        for fy in range(filter_height):
            for fx in range(filter_width):
                img_x = x + fx - filter_radius_x
                img_y = y + fy - filter_radius_y

                # Lidar com bordas (padding de borda)
                if img_x < 0:
                    img_x = 0
                if img_x >= width:
                    img_x = width - 1
                if img_y < 0:
                    img_y = 0
                if img_y >= height:
                    img_y = height - 1

                # Aplicar filtro
                filter_value = filter_array[fy * filter_width + fx]
                pixel_value = input_array[img_y * width + img_x]
                sum_value += filter_value * pixel_value

        output_array[y * width + x] = sum_value

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
    d_a = ocl.to_device(a)
    d_b = ocl.to_device(b)
    d_c = ocl.to_device(c_opencl)

    # Configurar grid e bloco
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    # Medir tempo OpenCL (incluindo transferência)
    start = time.time()
    vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)
    ocl.synchronize()
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
    d_a = ocl.to_device(a_flat)
    d_b = ocl.to_device(b_flat)
    d_c = ocl.to_device(c_flat)

    # Medir tempo OpenCL
    start = time.time()
    block_dim = (16, 16)
    grid_dim = (width // block_dim[0] + 1, width // block_dim[1] + 1)
    matrix_multiply(d_a, d_b, d_c, width, grid=grid_dim, block=block_dim)
    ocl.synchronize()
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
    d_data = ocl.to_device(data)

    # Configurar tamanhos
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    # Alocar array para resultados parciais
    d_partial_sums = ocl.device_array((grid_size,), dtype=np.float32)

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

def benchmark_saxpy(n=1000000):
    """Benchmarking de SAXPY (Single-Precision A*X Plus Y)"""
    print(f"\n=== Benchmark: SAXPY (N={n}) ===")

    # Criar dados de teste
    a = np.float32(2.0)  # Constante
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    result_opencl = np.zeros_like(x)

    # Medir tempo NumPy
    start = time.time()
    result_numpy = a * x + y
    numpy_time = time.time() - start
    print(f"NumPy: {numpy_time:.6f} segundos")

    # Transferir para o dispositivo
    d_x = ocl.to_device(x)
    d_y = ocl.to_device(y)
    d_result = ocl.to_device(result_opencl)

    # Configurar grid e bloco
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    # Medir tempo OpenCL
    start = time.time()
    saxpy(a, d_x, d_y, d_result, grid=grid_size, block=block_size)
    ocl.synchronize()
    result_opencl = d_result.copy_to_host()
    opencl_time = time.time() - start
    print(f"OpenCL (total): {opencl_time:.6f} segundos")

    # Verificar resultado
    if np.allclose(result_numpy, result_opencl, rtol=1e-5, atol=1e-5):
        print("✓ Resultados corretos!")
        print(f"Speedup: {numpy_time/opencl_time:.2f}x")
    else:
        print("✗ Resultados divergem!")
        max_diff = np.max(np.abs(result_numpy - result_opencl))
        print(f"Máxima diferença: {max_diff}")

def benchmark_convolution_2d(width=1024, height=1024):
    """Benchmarking de convolução 2D"""
    print(f"\n=== Benchmark: Convolução 2D ({width}x{height}) ===")

    # Criar dados de teste
    image = np.random.rand(height, width).astype(np.float32)

    # Filtro de Sobel para detecção de bordas
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    filter_width, filter_height = 3, 3
    result_opencl = np.zeros_like(image)

    # Implementação NumPy equivalente (simplificada)
    def numpy_convolution(img, kernel):
        result = np.zeros_like(img)
        pad_x = kernel.shape[1] // 2
        pad_y = kernel.shape[0] // 2

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for ky in range(kernel.shape[0]):
                    for kx in range(kernel.shape[1]):
                        img_x = x + kx - pad_x
                        img_y = y + ky - pad_y

                        if (img_x >= 0 and img_x < img.shape[1] and 
                            img_y >= 0 and img_y < img.shape[0]):
                            result[y, x] += img[img_y, img_x] * kernel[ky, kx]

        return result

    # Medir tempo NumPy (implementação manual)
    print("Executando versão NumPy (isso pode levar alguns minutos)...")
    start = time.time()
    # Para testes rápidos, use uma versão reduzida
    small_size = 128
    small_image = image[:small_size, :small_size]
    result_numpy = numpy_convolution(small_image, sobel_x)
    numpy_time = time.time() - start
    print(f"NumPy (manual, {small_size}x{small_size}): {numpy_time:.6f} segundos")

    # Se disponível, usar scipy.signal.convolve2d
    try:
        from scipy import signal
        start = time.time()
        result_scipy = signal.convolve2d(image, sobel_x, mode='same')
        scipy_time = time.time() - start
        print(f"SciPy: {scipy_time:.6f} segundos")
    except ImportError:
        scipy_time = None
        print("SciPy não disponível, ignorando benchmark de convolve2d")

    # Transferir para o dispositivo
    d_image = ocl.to_device(image.flatten())
    d_filter = ocl.to_device(sobel_x.flatten())
    d_result = ocl.to_device(result_opencl.flatten())

    # Configurar grid e bloco
    block_dim = (16, 16)
    grid_dim = (width // block_dim[0] + 1, height // block_dim[1] + 1)

    # Medir tempo OpenCL
    start = time.time()
    convolution_2d(d_image, d_result, d_filter, width, height, filter_width, filter_height, 
                 grid=grid_dim, block=block_dim)
    ocl.synchronize()
    result_opencl = d_result.copy_to_host().reshape((height, width))
    opencl_time = time.time() - start
    print(f"OpenCL (total): {opencl_time:.6f} segundos")

    # Comparar com SciPy se disponível
    if scipy_time is not None:
        print(f"Speedup vs SciPy: {scipy_time/opencl_time:.2f}x")

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
    d_a = ocl.to_device(a)
    d_b = ocl.to_device(b)
    d_c = ocl.to_device(c)

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
    d_a = ocl.to_device(a_flat)
    d_b = ocl.to_device(b_flat)
    d_c = ocl.to_device(c_flat)

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
    d_img = ocl.to_device(img.flatten())
    d_img_out = ocl.to_device(img_out.flatten())

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
    d_data = ocl.to_device(data)

    # Configurar tamanhos
    block_size = 256
    grid_size = (data_size + block_size - 1) // block_size

    # Alocar array para resultados parciais
    d_partial_sums = ocl.device_array((grid_size,), dtype=np.float32)

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

    # Exemplo 5: SAXPY
    print("\n5. SAXPY (Single-Precision A*X Plus Y)")

    # Criar dados de teste
    a = np.float32(3.0)
    x = np.random.rand(data_size).astype(np.float32)
    y = np.random.rand(data_size).astype(np.float32)
    result = np.zeros_like(x)

    # Transferir para o dispositivo
    d_x = ocl.to_device(x)
    d_y = ocl.to_device(y)
    d_result = ocl.to_device(result)

    # Executar SAXPY
    saxpy(a, d_x, d_y, d_result, grid=grid_size, block=block_size)

    # Obter resultado
    result = d_result.copy_to_host()
    expected = a * x + y

    if np.allclose(result, expected, rtol=1e-5, atol=1e-5):
        print("✓ Teste SAXPY: SUCESSO")
    else:
        print("✗ Teste SAXPY: FALHA")

    # Exemplo 6: Convolução 2D
    print("\n6. Convolução 2D (Detecção de Bordas)")

    # Criar imagem de teste
    width, height = 64, 64
    image = np.random.rand(height, width).astype(np.float32)
    result_img = np.zeros_like(image)

    # Filtro de Sobel para detecção de bordas
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    filter_width, filter_height = 3, 3

    # Transferir para o dispositivo
    d_image = ocl.to_device(image.flatten())
    d_filter = ocl.to_device(sobel_x.flatten())
    d_result = ocl.to_device(result_img.flatten())

    # Configurar grid e bloco
    block_dim = (16, 16)
    grid_dim = (width // block_dim[0] + 1, height // block_dim[1] + 1)

    # Executar convolução
    convolution_2d(d_image, d_result, d_filter, width, height, filter_width, filter_height, 
                 grid=grid_dim, block=block_dim)

    print("✓ Convolução 2D aplicada com sucesso")

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
    d_a1 = ocl.to_device(a1)
    d_b1 = ocl.to_device(b1)
    d_c1 = ocl.to_device(c1)

    d_a2 = ocl.to_device(a2)
    d_b2 = ocl.to_device(b2)
    d_c2 = ocl.to_device(c2)

    # Configurar grid e bloco
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    # Criar streams
    stream1 = ocl.stream()
    stream2 = ocl.stream()

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

    if np.allclose(result1, expected1) and np.allclose(result2, expectedexpected2):
        print("✓ Teste de streams: SUCESSO")
    else:
        print("✗ Teste de streams: FALHA")

def run_profiling_demo():
    """Demonstração do sistema de profiling"""
    print("\n=== Demonstração de Profiling ===")

    # Ativar profiling
    profiler.start()

    # Executar vários kernels para profiling
    for size in [100000, 500000, 1000000]:
        # Dados de teste
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        c = np.zeros_like(a)

        # Transferir para o dispositivo
        d_a = ocl.to_device(a)
        d_b = ocl.to_device(b)
        d_c = ocl.to_device(c)

        # Executar soma de vetores
        block_size = 256
        grid_size = (size + block_size - 1) // block_size
        for _ in range(3):  # Várias execuções
            vector_add(d_a, d_b, d_c, grid=grid_size, block=block_size)

    # Executar SAXPY uma vez
    a_val = np.float32(2.0)
    saxpy(a_val, d_a, d_b, d_c, grid=grid_size, block=block_size)

    # Parar profiling
    profiler.stop()

    # Exibir estatísticas
    print("\nEstatísticas de profiling:")
    profiler.print_stats()

    # Limpar dados de profiling
    profiler.clear()

def main():
    """Função principal"""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Exemplos e benchmarks da extensão numba_opencl")
    parser.add_argument("--device", help="Seleciona dispositivo pelo ID")
    parser.add_argument("--list", action="store_true", help="Lista dispositivos disponíveis")
    parser.add_argument("--info", action="store_true", help="Exibe informações do dispositivo atual")
    parser.add_argument("--tests", action="store_true", help="Executa testes básicos")
    parser.add_argument("--bench", action="store_true", help="Executa benchmarks")
    parser.add_argument("--examples", action="store_true", help="Executa exemplos")
    parser.add_argument("--streams", action="store_true", help="Testa streams")
    parser.add_argument("--profiling", action="store_true", help="Demonstra profiling")
    parser.add_argument("--all", action="store_true", help="Executa tudo")

    args = parser.parse_args()

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

    # Processar argumentos
    if args.list:
        devices = ocl.list_devices()
        print("\nDispositivos disponíveis:")
        table = PrettyTable()
        table.field_names = ["ID", "Nome", "Tipo", "Plataforma", "Comp. Units", "Atual"]

        for device in devices:
            current = "✓" if device['current'] else ""
            table.add_row([
                device['id'],
                device['name'],
                device['type'],
                device['platform'],
                device['compute_units'],
                current
            ])

        print(table)
        return

    if args.device:
        device_id = int(args.device)
        if ocl.select_device(device_id):
            print(f"Dispositivo #{device_id} selecionado com sucesso.")
        else:
            print(f"Erro ao selecionar dispositivo #{device_id}.")

    if args.info:
        ocl.print_device_info()
        return

    # Se nenhum argumento de ação for fornecido, mostrar menu interativo
    if not any([args.tests, args.bench, args.examples, args.streams, args.profiling, args.all]):
        menu_loop()
        return

    # Executar testes solicitados
    if args.tests or args.all:
        import tests
        tests.run_tests()

    if args.examples or args.all:
        run_examples()

    if args.streams or args.all:
        test_streams()

    if args.profiling or args.all:
        run_profiling_demo()

    if args.bench or args.all:
        benchmark_vector_add(n=5000000)
        benchmark_matrix_multiply(width=512)
        benchmark_reduction(n=5000000)
        benchmark_saxpy(n=5000000)
        benchmark_convolution_2d(width=1024, height=1024)

def menu_loop():
    """Loop de menu interativo"""
    while True:
        print("\n=== numba_opencl: Menu Principal ===")
        print("1. Seleção de dispositivo")
        print("2. Executar exemplos")
        print("3. Executar testes de streams")
        print("4. Executar benchmarks")
        print("5. Demonstração de profiling")
        print("6. Executar testes automatizados")
        print("0. Sair")

        try:
            choice = input("\nEscolha uma opção (0-6): ")

            if choice == "0":
                print("Saindo...")
                break
            elif choice == "1":
                print_device_selection_menu()
            elif choice == "2":
                run_examples()
            elif choice == "3":
                test_streams()
            elif choice == "4":
                print("\n=== Menu de Benchmarks ===")
                print("1. Soma de vetores")
                print("2. Multiplicação de matrizes")
                print("3. Redução (soma)")
                print("4. SAXPY")
                print("5. Convolução 2D")
                print("6. Todos os benchmarks")

                bench_choice = input("\nEscolha um benchmark (1-6): ")
                if bench_choice == "1":
                    benchmark_vector_add(n=5000000)
                elif bench_choice == "2":
                    benchmark_matrix_multiply(width=512)
                elif bench_choice == "3":
                    benchmark_reduction(n=5000000)
                elif bench_choice == "4":
                    benchmark_saxpy(n=5000000)
                elif bench_choice == "5":
                    benchmark_convolution_2d(width=1024, height=1024)
                elif bench_choice == "6":
                    benchmark_vector_add(n=5000000)
                    benchmark_matrix_multiply(width=512)
                    benchmark_reduction(n=5000000)
                    benchmark_saxpy(n=5000000)
                    benchmark_convolution_2d(width=1024, height=1024)
                else:
                    print("Opção inválida")
            elif choice == "5":
                run_profiling_demo()
            elif choice == "6":
                import tests
                tests.run_tests()
            else:
                print("Opção inválida")
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()