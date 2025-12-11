"""
Demo 2 Enhanced: Background Color Detection with Gradient Support

This demo demonstrates:
1. Solid color detection (baseline)
2. Gradient color detection with multiple strategies
3. Comparison of detection methods: Mode, Mean, Median
4. Visual comparison of results
"""

import fitz  # PyMuPDF
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw
import io


def detect_color_mode(pixels: np.ndarray) -> tuple:
    """
    Detect color using MODE (most frequent color).
    Current algorithm - good for solid colors.

    Args:
        pixels: Numpy array of RGB pixels, shape (n, 3)

    Returns:
        RGB tuple (r, g, b)
    """
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixel_tuples)
    most_common = color_counts.most_common(1)[0][0]
    return most_common


def detect_color_mean(pixels: np.ndarray) -> tuple:
    """
    Detect color using MEAN (average color).
    Good for gradients - returns midpoint color.

    Args:
        pixels: Numpy array of RGB pixels, shape (n, 3)

    Returns:
        RGB tuple (r, g, b)
    """
    mean_color = np.mean(pixels, axis=0).astype(int)
    return tuple(mean_color)


def detect_color_median(pixels: np.ndarray) -> tuple:
    """
    Detect color using MEDIAN (middle value).
    Good for gradients with outliers.

    Args:
        pixels: Numpy array of RGB pixels, shape (n, 3)

    Returns:
        RGB tuple (r, g, b)
    """
    median_color = np.median(pixels, axis=0).astype(int)
    return tuple(median_color)


def detect_background_color_multi(page: fitz.Page, bbox: tuple) -> dict:
    """
    Detect background color using multiple strategies.

    Args:
        page: PyMuPDF page object
        bbox: Tuple of (x0, y0, x1, y1) coordinates

    Returns:
        Dict with results from different strategies
    """
    # Create a rect from bbox
    rect = fitz.Rect(bbox)

    # Get pixmap
    mat = fitz.Matrix(150/72, 150/72)
    pix = page.get_pixmap(matrix=mat, clip=rect)

    # Convert to numpy array
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.n == 3:  # RGB
        img_array = img_data.reshape(pix.height, pix.width, 3)
    elif pix.n == 4:  # RGBA
        img_array = img_data.reshape(pix.height, pix.width, 4)[:, :, :3]
    else:
        img_array = img_data.reshape(pix.height, pix.width, pix.n)

    # Sample border pixels (current algorithm)
    border_width = max(2, min(pix.width, pix.height) // 10)
    border_pixels = []
    border_pixels.append(img_array[:border_width, :, :].reshape(-1, 3))
    border_pixels.append(img_array[-border_width:, :, :].reshape(-1, 3))
    border_pixels.append(img_array[:, :border_width, :].reshape(-1, 3))
    border_pixels.append(img_array[:, -border_width:, :].reshape(-1, 3))
    all_border_pixels = np.vstack(border_pixels)

    # Also get ALL pixels for comparison
    all_pixels = img_array.reshape(-1, 3)

    # Detect using different strategies
    results = {
        'mode_border': detect_color_mode(all_border_pixels),
        'mode_all': detect_color_mode(all_pixels),
        'mean_border': detect_color_mean(all_border_pixels),
        'mean_all': detect_color_mean(all_pixels),
        'median_border': detect_color_median(all_border_pixels),
        'median_all': detect_color_median(all_pixels),
    }

    return results


def create_gradient_image(width, height, start_color, end_color, direction='horizontal'):
    """
    Create a gradient image using PIL.

    Args:
        width, height: Image dimensions
        start_color: Starting RGB tuple (0-255)
        end_color: Ending RGB tuple (0-255)
        direction: 'horizontal' or 'vertical'

    Returns:
        PIL Image object
    """
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    if direction == 'horizontal':
        for x in range(width):
            ratio = x / width
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            draw.line([(x, 0), (x, height)], fill=(r, g, b))
    else:  # vertical
        for y in range(height):
            ratio = y / height
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img


def insert_gradient_in_pdf(page, rect, start_color, end_color, direction='horizontal'):
    """
    Insert a gradient into a PDF page.

    Args:
        page: PyMuPDF page
        rect: fitz.Rect for gradient position
        start_color, end_color: RGB tuples (0-255)
        direction: 'horizontal' or 'vertical'
    """
    width = int(rect.width * 2)  # Higher resolution
    height = int(rect.height * 2)

    # Create gradient image
    gradient_img = create_gradient_image(width, height, start_color, end_color, direction)

    # Convert to bytes
    img_buffer = io.BytesIO()
    gradient_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    # Insert into PDF
    page.insert_image(rect, stream=img_buffer)


def create_test_pdf():
    """
    Create test PDF with solid colors and gradients.
    """
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)  # Landscape A4

    # Title
    title_rect = fitz.Rect(50, 20, 792, 50)
    page.insert_textbox(
        title_rect,
        "Gradient Color Detection Test - Strategy Comparison",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    # Test regions
    test_regions = []

    # Section 1: Solid Colors (baseline)
    y_offset = 70
    page.insert_text((50, y_offset), "Solid Colors (Baseline):", fontsize=12, fontname="helv")
    y_offset += 20

    solid_tests = [
        ("Red", (255, 0, 0)),
        ("Blue", (0, 0, 255)),
        ("Green", (0, 255, 0)),
    ]

    x = 50
    for name, color in solid_tests:
        rect = fitz.Rect(x, y_offset, x + 100, y_offset + 60)
        # Draw solid color
        color_norm = tuple(c / 255 for c in color)
        page.draw_rect(rect, color=color_norm, fill=color_norm)
        # Label
        page.insert_text((x + 5, y_offset + 75), name, fontsize=10, fontname="helv")
        test_regions.append((name, rect, "solid", color))
        x += 120

    # Section 2: Linear Gradients
    y_offset = 180
    page.insert_text((50, y_offset), "Linear Gradients:", fontsize=12, fontname="helv")
    y_offset += 20

    gradient_tests = [
        ("Red→Blue (H)", (255, 0, 0), (0, 0, 255), "horizontal"),
        ("White→Black (H)", (255, 255, 255), (0, 0, 0), "horizontal"),
        ("Yellow→Orange (V)", (255, 255, 0), (255, 128, 0), "vertical"),
        ("Green→Cyan (V)", (0, 255, 0), (0, 255, 255), "vertical"),
    ]

    x = 50
    for name, start_color, end_color, direction in gradient_tests:
        rect = fitz.Rect(x, y_offset, x + 100, y_offset + 80)
        # Draw gradient
        insert_gradient_in_pdf(page, rect, start_color, end_color, direction)
        # Label
        page.insert_text((x + 5, y_offset + 90), name, fontsize=9, fontname="helv")
        test_regions.append((name, rect, "gradient", (start_color, end_color, direction)))
        x += 120

    # Save test PDF
    test_pdf_path = "output/test_gradients.pdf"
    doc.save(test_pdf_path)
    doc.close()

    return test_pdf_path, test_regions


def test_gradient_detection():
    """
    Test color detection on gradients with multiple strategies.
    """
    print("\n" + "=" * 80)
    print("GRADIENT COLOR DETECTION TEST")
    print("=" * 80)

    # Create test PDF
    pdf_path, test_regions = create_test_pdf()
    print(f"\n[Created] Test PDF: {pdf_path}")

    # Open PDF for detection
    doc = fitz.open(pdf_path)
    page = doc[0]

    # Test each region and collect results
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)

    all_results = []

    for name, rect, region_type, color_info in test_regions:
        print(f"\n{'─' * 80}")
        print(f"Region: {name} ({region_type})")
        print(f"{'─' * 80}")

        # Detect with all strategies
        results = detect_background_color_multi(page, rect)

        # Print results
        print(f"{'Strategy':<20} {'RGB Value':<25} {'Description':<30}")
        print(f"{'-' * 80}")
        print(f"{'Mode (border)':<20} {str(results['mode_border']):<25} {'Most frequent at edges':<30}")
        print(f"{'Mode (all)':<20} {str(results['mode_all']):<25} {'Most frequent overall':<30}")
        print(f"{'Mean (border)':<20} {str(results['mean_border']):<25} {'Average of edges':<30}")
        print(f"{'Mean (all)':<20} {str(results['mean_all']):<25} {'Average overall':<30}")
        print(f"{'Median (border)':<20} {str(results['median_border']):<25} {'Middle value at edges':<30}")
        print(f"{'Median (all)':<20} {str(results['median_all']):<25} {'Middle value overall':<30}")

        # For gradients, calculate expected average
        if region_type == "gradient":
            start, end, direction = color_info
            expected_avg = tuple((s + e) // 2 for s, e in zip(start, end))
            print(f"\n{'Expected midpoint:':<20} {str(expected_avg):<25} {'(theoretical average)':<30}")

        # Store results for visualization
        all_results.append((name, rect, region_type, color_info, results))

    # Create visualization PDF (pass open document and results)
    create_visualization_pdf(doc, all_results)

    doc.close()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\nFor SOLID colors:")
    print("  - Use: Mode (border) - Fast and accurate")
    print("\nFor GRADIENTS:")
    print("  - Use: Mean (all) - Represents average/midpoint color")
    print("  - Alternative: Median (all) - Robust to outliers")
    print("\nFor MIXED content (text over gradient):")
    print("  - Use: Mean (border) - Avoids text, gets background average")
    print("=" * 80 + "\n")


def create_visualization_pdf(doc, all_results):
    """
    Create visualization showing detected colors.

    Args:
        doc: Open PyMuPDF document
        all_results: List of tuples (name, rect, region_type, color_info, results)
    """
    # Create new page for results
    result_page = doc.new_page(width=842, height=595)

    # Title
    title_rect = fitz.Rect(50, 20, 792, 50)
    result_page.insert_textbox(
        title_rect,
        "Detected Colors Visualization",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    # For each region, show detected colors
    y_offset = 70
    for name, rect, region_type, color_info, results in all_results[:7]:  # Limit to fit page
        # Label
        result_page.insert_text((50, y_offset), f"{name}:", fontsize=11, fontname="helv")

        # Show different strategies
        x = 200
        strategies = [
            ('Mode', results['mode_all']),
            ('Mean', results['mean_all']),
            ('Median', results['median_all']),
        ]

        for strategy_name, color_rgb in strategies:
            # Convert numpy types to regular ints
            color_rgb_clean = tuple(int(c) for c in color_rgb)

            # Draw color block
            color_rect = fitz.Rect(x, y_offset - 10, x + 60, y_offset + 20)
            color_norm = tuple(c / 255 for c in color_rgb_clean)
            result_page.draw_rect(color_rect, color=color_norm, fill=color_norm)
            result_page.draw_rect(color_rect, color=(0, 0, 0), width=0.5)

            # Label strategy
            result_page.insert_text((x + 2, y_offset + 30), strategy_name, fontsize=8, fontname="helv")

            x += 80

        y_offset += 50

    # Save
    output_path = "output/demo_color_detect_gradient.pdf"
    doc.save(output_path)

    print(f"\n[SUCCESS] Visualization PDF created: {output_path}")
    print("   - Page 1: Original test patterns")
    print("   - Page 2: Detected colors comparison")


if __name__ == "__main__":
    test_gradient_detection()
