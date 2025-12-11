"""
Performance Test: Color Detection Methods
测试三种颜色检测方法的性能
"""

import fitz
import numpy as np
from collections import Counter
import time


def detect_color_mode(pixels: np.ndarray) -> tuple:
    """Mode strategy - most frequent color"""
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixel_tuples)
    most_common = color_counts.most_common(1)[0][0]
    return most_common


def detect_color_mean(pixels: np.ndarray) -> tuple:
    """Mean strategy - average color"""
    mean_color = np.mean(pixels, axis=0).astype(int)
    return tuple(mean_color)


def detect_color_median(pixels: np.ndarray) -> tuple:
    """Median strategy - middle value"""
    median_color = np.median(pixels, axis=0).astype(int)
    return tuple(median_color)


def get_pixels_from_bbox(page: fitz.Page, bbox: tuple, sample_type: str = "border"):
    """Extract pixels from bbox"""
    rect = fitz.Rect(bbox)
    mat = fitz.Matrix(150/72, 150/72)
    pix = page.get_pixmap(matrix=mat, clip=rect)

    img_data = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.n == 3:
        img_array = img_data.reshape(pix.height, pix.width, 3)
    elif pix.n == 4:
        img_array = img_data.reshape(pix.height, pix.width, 4)[:, :, :3]
    else:
        img_array = img_data.reshape(pix.height, pix.width, pix.n)

    if sample_type == "border":
        border_width = max(2, min(pix.width, pix.height) // 10)
        border_pixels = []
        border_pixels.append(img_array[:border_width, :, :].reshape(-1, 3))
        border_pixels.append(img_array[-border_width:, :, :].reshape(-1, 3))
        border_pixels.append(img_array[:, :border_width, :].reshape(-1, 3))
        border_pixels.append(img_array[:, -border_width:, :].reshape(-1, 3))
        return np.vstack(border_pixels)
    else:  # all
        return img_array.reshape(-1, 3)


def performance_test():
    """Test performance of three detection methods"""

    print("\n" + "=" * 80)
    print("颜色检测方法性能测试")
    print("=" * 80)

    # Open test PDF
    pdf_path = "output/test_gradients.pdf"
    doc = fitz.open(pdf_path)
    page = doc[0]

    # Test regions (solid color and gradient)
    test_regions = [
        ("纯色 (Red)", (50, 90, 150, 150)),
        ("渐变 (Red->Blue)", (50, 200, 150, 280)),
    ]

    num_runs = 100  # 运行次数

    for region_name, bbox in test_regions:
        print(f"\n{'─' * 80}")
        print(f"测试区域: {region_name}")
        print(f"{'─' * 80}")

        for sample_type in ["border", "all"]:
            print(f"\n采样方式: {sample_type}")

            # Get pixels once (shared for all methods)
            pixels = get_pixels_from_bbox(page, bbox, sample_type)
            pixel_count = len(pixels)
            print(f"像素数量: {pixel_count:,}")

            # Test Mode
            times_mode = []
            for _ in range(num_runs):
                start = time.perf_counter()
                result = detect_color_mode(pixels)
                end = time.perf_counter()
                times_mode.append(end - start)
            avg_mode = np.mean(times_mode) * 1000  # Convert to ms
            std_mode = np.std(times_mode) * 1000

            # Test Mean
            times_mean = []
            for _ in range(num_runs):
                start = time.perf_counter()
                result = detect_color_mean(pixels)
                end = time.perf_counter()
                times_mean.append(end - start)
            avg_mean = np.mean(times_mean) * 1000
            std_mean = np.std(times_mean) * 1000

            # Test Median
            times_median = []
            for _ in range(num_runs):
                start = time.perf_counter()
                result = detect_color_median(pixels)
                end = time.perf_counter()
                times_median.append(end - start)
            avg_median = np.mean(times_median) * 1000
            std_median = np.std(times_median) * 1000

            # Print results
            print(f"\n{'方法':<15} {'平均耗时 (ms)':<20} {'标准差 (ms)':<20} {'相对速度':<15}")
            print(f"{'-' * 70}")

            # Calculate relative speed (Mode as baseline)
            print(f"{'Mode':<15} {avg_mode:>8.4f}            {std_mode:>8.4f}            {'1.00x (基准)':<15}")
            print(f"{'Mean':<15} {avg_mean:>8.4f}            {std_mean:>8.4f}            {avg_mean/avg_mode:>6.2f}x")
            print(f"{'Median':<15} {avg_median:>8.4f}            {std_median:>8.4f}            {avg_median/avg_mode:>6.2f}x")

    doc.close()

    print("\n" + "=" * 80)
    print("性能测试总结")
    print("=" * 80)
    print("\n预期结果：")
    print("  - Mean (平均): 最快 - 使用numpy向量化运算")
    print("  - Median (中位数): 中等 - numpy排序操作")
    print("  - Mode (众数): 最慢 - 需要遍历和计数")
    print("\n建议：")
    print("  - 纯色背景: 使用 Mode (准确性更重要)")
    print("  - 渐变背景: 使用 Mean (速度快且准确)")
    print("  - 性能优先: 使用 Mean (比 Mode 快数倍)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    performance_test()
