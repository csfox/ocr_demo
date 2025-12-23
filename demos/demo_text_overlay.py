"""
Demo 3: Text Overlay from JSON with Letter Replacement

This demo demonstrates:
1. Reading PDF and OCR JSON via command-line arguments
2. Detecting background color for each element (using Mean strategy)
3. Replacing all letters with 'c' while preserving formulas and spaces
4. Rendering back to PDF with adaptive font sizing and inline LaTeX
"""

import fitz  # PyMuPDF
import json
import argparse
import numpy as np
import re
import math
import time
import base64
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import io
from PIL import Image
from weasyprint import HTML
from json_translator import translate_element_text, should_skip_translation as should_skip_translation_check
import httpx
from openai import OpenAI


# ==================== OCR 服务配置（对齐 local/conf/service.yaml） ====================
OCR_API_BASE_URL = "http://127.0.0.1:9000/v1"  # 本地 OCR 服务地址
OCR_API_KEY = "sk-xxxxxxxxxxxx"                 # API Key
OCR_MODEL = "model"                             # 模型名称

# 超时配置（单位：秒）
OCR_TIMEOUT = 180

# 推理参数
OCR_TEMPERATURE = 0.1
OCR_TOP_P = 1.0
OCR_MAX_COMPLETION_TOKENS = 16384

# 图片尺寸常量（对齐 Dots OCR）
IMAGE_FACTOR = 28
MIN_PIXELS = 3136           # 56x56
MAX_PIXELS = 11289600       # 3360x3360

# Prompt 定义（完整版，一行都不少）
OCR_PROMPT_LAYOUT_ALL = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""


# Language-specific line height map (from restorer.py)
LANG_LINEHEIGHT_MAP = {
    "zh-cn": 1.4,  # Chinese
    "ja": 1.1,     # Japanese
    "ko": 1.2,     # Korean
    "en": 1.2,     # English
    "ar": 1.0,     # Arabic
    "ru": 0.8,     # Russian
}


# ==================== 图片尺寸计算（对齐 Dots OCR） ====================

def round_by_factor(value: int, factor: int) -> int:
    """四舍五入到最近的 factor 倍数"""
    return ((value + factor // 2) // factor) * factor


def floor_by_factor(value: int, factor: int) -> int:
    """向下取整到 factor 倍数"""
    return (value // factor) * factor


def ceil_by_factor(value: int, factor: int) -> int:
    """向上取整到 factor 倍数"""
    return ((value + factor - 1) // factor) * factor


def calculate_target_size(height: int, width: int) -> tuple:
    """
    计算目标尺寸（对齐 Dots OCR 的 Python 版本逻辑）

    Args:
        height: 原始高度
        width: 原始宽度

    Returns:
        (target_height, target_width)
    """
    # 1. 验证长宽比
    max_dim = max(height, width)
    min_dim = min(height, width)
    if min_dim == 0:
        raise ValueError(f"Invalid dimensions: height={height}, width={width}")
    if max_dim / min_dim > 200:
        raise ValueError(f"Aspect ratio {max_dim / min_dim:.2f} exceeds maximum 200")

    # 2. 先对原始尺寸进行 round 对齐，并确保至少为 IMAGE_FACTOR
    h_bar = max(IMAGE_FACTOR, round_by_factor(height, IMAGE_FACTOR))
    w_bar = max(IMAGE_FACTOR, round_by_factor(width, IMAGE_FACTOR))

    # 3. 如果对齐后超出 MAX_PIXELS：使用原始尺寸重新计算（floor）
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h_bar = max(IMAGE_FACTOR, floor_by_factor(int(height / beta), IMAGE_FACTOR))
        w_bar = max(IMAGE_FACTOR, floor_by_factor(int(width / beta), IMAGE_FACTOR))
    elif h_bar * w_bar < MIN_PIXELS:
        # 4. 如果对齐后低于 MIN_PIXELS：使用原始尺寸重新计算（ceil）
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = ceil_by_factor(int(height * beta), IMAGE_FACTOR)
        w_bar = ceil_by_factor(int(width * beta), IMAGE_FACTOR)

        # 5. 二次检查是否超出 MAX_PIXELS
        if h_bar * w_bar > MAX_PIXELS:
            beta = math.sqrt(h_bar * w_bar / MAX_PIXELS)
            h_bar = max(IMAGE_FACTOR, floor_by_factor(int(h_bar / beta), IMAGE_FACTOR))
            w_bar = max(IMAGE_FACTOR, floor_by_factor(int(w_bar / beta), IMAGE_FACTOR))

    return h_bar, w_bar


def resize_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    调整图片尺寸以符合 Dots OCR 要求

    Args:
        img: PIL Image 对象

    Returns:
        调整后的 PIL Image 对象
    """
    width, height = img.size
    target_h, target_w = calculate_target_size(height, width)

    if target_h != height or target_w != width:
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return img


def image_to_base64(img: Image.Image) -> str:
    """
    将 PIL Image 转换为 base64 字符串

    Args:
        img: PIL Image 对象

    Returns:
        base64 编码的字符串
    """
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ==================== PDF 转图片 ====================

def pdf_page_to_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    """
    将 PDF 页面转换为 PIL Image。

    Args:
        page: PyMuPDF 页面对象
        dpi: 图片 DPI（默认 200）

    Returns:
        PIL Image 对象
    """
    # 计算缩放因子（PDF 默认 72 DPI）
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # 渲染页面为 pixmap
    pix = page.get_pixmap(matrix=mat)

    # 转换为 PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    return img


# ==================== OCR API 调用 ====================

def call_ocr_api(img: Image.Image, page_index: int,
                 retry_count: int = 3, retry_delay_ms: int = 1000) -> dict:
    """
    调用 OCR API 提取图片内容结构（使用 OpenAI 兼容格式）。

    Args:
        img: PIL Image 对象
        page_index: 页面索引（用于日志）
        retry_count: 重试次数（默认 3）
        retry_delay_ms: 重试延迟毫秒（默认 1000）

    Returns:
        OCR 结果字典，包含 elements 列表

    Raises:
        Exception: OCR 请求失败
    """
    # 获取原始图片尺寸
    orig_width, orig_height = img.size

    # 调整图片尺寸
    resized_img = resize_image_for_ocr(img)
    new_width, new_height = resized_img.size

    # 转换为 base64
    base64_img = image_to_base64(resized_img)

    # 构建完整 prompt（加上 Dots OCR 特殊前缀）
    full_prompt = "<|img|><|imgpad|><|endofimg|>" + OCR_PROMPT_LAYOUT_ALL

    # 创建 OpenAI 客户端（禁用代理，设置超时）
    # trust_env=False 完全忽略环境变量中的代理设置
    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(float(OCR_TIMEOUT), connect=30.0),
    )
    client = OpenAI(
        api_key=OCR_API_KEY,
        base_url=OCR_API_BASE_URL,
        http_client=http_client,
    )

    for attempt in range(1, retry_count + 1):
        try:
            print(f"    [OCR] 页面 {page_index + 1}: 第 {attempt}/{retry_count} 次请求...")
            print(f"    [OCR] 图片尺寸: {orig_width}x{orig_height} -> {new_width}x{new_height}")

            # 调用 OpenAI 兼容 API（参数对齐 local/conf/service.yaml）
            response = client.chat.completions.create(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_img}"
                                }
                            },
                            {
                                "type": "text",
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                max_tokens=OCR_MAX_COMPLETION_TOKENS,
                temperature=OCR_TEMPERATURE,
                top_p=OCR_TOP_P,
            )

            # 解析响应
            content = response.choices[0].message.content
            if not content:
                raise Exception("OCR 响应内容为空")

            # 解析 JSON
            try:
                elements = json.loads(content)
            except json.JSONDecodeError:
                # 尝试清理响应（移除可能的 markdown 代码块标记）
                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                elements = json.loads(cleaned.strip())

            print(f"    [OCR] 页面 {page_index + 1}: 成功，识别 {len(elements)} 个元素")

            # 检查第一个元素的 bbox 格式
            if elements:
                sample_bbox = elements[0].get("bbox", [])
                print(f"    [OCR] 样本 bbox: {sample_bbox} (基于 {new_width}x{new_height})")

            # OCR 返回的 bbox 是像素坐标，基于 resize 后的图片尺寸
            # 需要转换回原始图片尺寸
            scale_x = orig_width / new_width
            scale_y = orig_height / new_height

            converted_elements = []
            for elem in elements:
                bbox = elem.get("bbox", [0, 0, 0, 0])

                # 将 bbox 从 resize 后尺寸转换为原始尺寸
                pixel_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                ]

                converted_elements.append({
                    "bbox": pixel_bbox,
                    "category": elem.get("category", "Text"),
                    "text": elem.get("text", ""),
                })

            return {
                "page_index": page_index,
                "elements": converted_elements,
                "image_info": {
                    "width": orig_width,
                    "height": orig_height,
                },
            }

        except Exception as e:
            print(f"    [OCR] 页面 {page_index + 1}: 第 {attempt} 次请求失败 - {e}")

            if attempt < retry_count:
                # 等待后重试（递增延迟）
                delay = retry_delay_ms * attempt / 1000.0
                time.sleep(delay)
            else:
                raise Exception(f"OCR 请求失败，已重试 {retry_count} 次: {e}")


def extract_pdf_with_ocr(pdf_path: str, image_dpi: int = 200) -> dict:
    """
    使用 OCR API 提取 PDF 所有页面的内容。

    Args:
        pdf_path: PDF 文件路径
        image_dpi: 图片 DPI（默认 200）

    Returns:
        包含所有页面元素的字典 {"pages": [...]}
    """
    print(f"\n[OCR] 开始提取 PDF: {pdf_path}")
    print(f"[OCR] 图片 DPI: {image_dpi}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[OCR] 共 {total_pages} 页")

    pages_data = []

    for page_index in range(total_pages):
        page = doc[page_index]
        print(f"\n[OCR] 处理页面 {page_index + 1}/{total_pages}...")

        # 转换页面为图片
        img = pdf_page_to_image(page, dpi=image_dpi)
        print(f"    [OCR] 原始图片尺寸: {img.size[0]}x{img.size[1]}")

        # 调用 OCR API
        page_result = call_ocr_api(img, page_index)
        pages_data.append(page_result)

    doc.close()

    print(f"\n[OCR] 提取完成，共 {len(pages_data)} 页")

    return {"pages": pages_data}


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.

    Args:
        text: Input text

    Returns:
        Language code ('zh-cn', 'en', etc.)
    """
    # Remove formulas for detection
    text_without_formulas = re.sub(r'\$(.+?)\$', '', text)

    # Count character types
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', text_without_formulas))
    latin_count = len(re.findall(r'[a-zA-Z]', text_without_formulas))

    # If CJK characters are significant, consider it Chinese
    if cjk_count > latin_count * 0.3:
        return "zh-cn"
    return "en"


def is_cjk_char(ch: str) -> bool:
    """
    Check if a character needs CJK font to render properly.
    Includes CJK ideographs, punctuation, symbols, and full-width characters.
    """
    if not ch:
        return False
    code = ord(ch)
    # CJK Unified Ideographs
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # CJK Unified Ideographs Extension A
    if 0x3400 <= code <= 0x4DBF:
        return True
    # CJK Unified Ideographs Extension B-F (rarely used)
    if 0x20000 <= code <= 0x2EBEF:
        return True
    # CJK Compatibility Ideographs
    if 0xF900 <= code <= 0xFAFF:
        return True
    # CJK Symbols and Punctuation (。、「」『』【】〈〉《》等)
    if 0x3000 <= code <= 0x303F:
        return True
    # Hiragana (Japanese)
    if 0x3040 <= code <= 0x309F:
        return True
    # Katakana (Japanese)
    if 0x30A0 <= code <= 0x30FF:
        return True
    # Hangul (Korean)
    if 0xAC00 <= code <= 0xD7AF:
        return True
    # Fullwidth ASCII variants (！＂＃＄％＆等) and Halfwidth/Fullwidth Forms
    if 0xFF00 <= code <= 0xFFEF:
        return True
    # General Punctuation (includes ·, –, —, ', ', ", " etc.)
    if 0x2000 <= code <= 0x206F:
        return True
    # CJK Radicals Supplement
    if 0x2E80 <= code <= 0x2EFF:
        return True
    # Kangxi Radicals
    if 0x2F00 <= code <= 0x2FDF:
        return True
    # Enclosed CJK Letters and Months
    if 0x3200 <= code <= 0x32FF:
        return True
    # CJK Compatibility
    if 0x3300 <= code <= 0x33FF:
        return True
    # Vertical Forms
    if 0xFE10 <= code <= 0xFE1F:
        return True
    # CJK Compatibility Forms
    if 0xFE30 <= code <= 0xFE4F:
        return True
    # Small Form Variants
    if 0xFE50 <= code <= 0xFE6F:
        return True
    # Halfwidth and Fullwidth Forms (extra)
    if 0xFFE0 <= code <= 0xFFEF:
        return True
    # Mathematical symbols that may need special handling (×, ÷, etc.)
    if code == 0x00D7 or code == 0x00F7:  # × ÷
        return True
    return False


def detect_color_mean(pixels: np.ndarray) -> tuple:
    """
    Detect color using MEAN (average color).
    Fast and accurate for both solid colors and gradients.

    Args:
        pixels: Numpy array of RGB pixels, shape (n, 3)

    Returns:
        RGB tuple (r, g, b)
    """
    mean_color = np.mean(pixels, axis=0).astype(int)
    return tuple(mean_color)


def detect_background_color(page: fitz.Page, bbox: tuple, border_width: int = 1) -> tuple:
    """
    Detect background color from the border area OUTSIDE the bbox.

    Args:
        page: PyMuPDF page object
        bbox: Tuple of (x0, y0, x1, y1) coordinates
        border_width: Width of border area to sample (default: 1 pixels)

    Returns:
        RGB tuple (r, g, b), defaults to white (255, 255, 255) on error
    """
    try:
        x0, y0, x1, y1 = bbox

        # Expand bbox to include surrounding border
        expanded_rect = fitz.Rect(
            x0 - border_width,
            y0 - border_width,
            x1 + border_width,
            y1 + border_width
        )

        # Clip to page boundaries
        page_rect = page.rect
        expanded_rect = expanded_rect & page_rect

        # Check if rect is valid
        if expanded_rect.is_empty or expanded_rect.width < 1 or expanded_rect.height < 1:
            print(f"  [WARNING] Invalid bbox: {bbox}, using white background")
            return (255, 255, 255)

        # Get pixmap of expanded area
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat, clip=expanded_rect)

        # Check if pixmap has data
        if pix.width < 1 or pix.height < 1:
            print(f"  [WARNING] Empty pixmap for bbox: {bbox}, using white background")
            return (255, 255, 255)

        # Convert to numpy array
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)

        if pix.n == 3:  # RGB
            img_array = img_data.reshape(pix.height, pix.width, 3)
        elif pix.n == 4:  # RGBA
            img_array = img_data.reshape(pix.height, pix.width, 4)[:, :, :3]
        else:
            img_array = img_data.reshape(pix.height, pix.width, pix.n)

        # Calculate border dimensions in the pixmap
        # Scale border_width according to DPI ratio (150/72)
        scaled_border = int(border_width * 150 / 72)

        # Sample only LEFT and RIGHT border areas (not top/bottom)
        # Left border (full height)
        left_border = img_array[:, :scaled_border, :].reshape(-1, 3)
        # Right border (full height)
        right_border = img_array[:, -scaled_border:, :].reshape(-1, 3)

        # Combine left and right border pixels only
        border_pixels = np.vstack([left_border, right_border])

        # Check if we have pixels
        if len(border_pixels) == 0:
            print(f"  [WARNING] No pixels sampled for bbox: {bbox}, using white background")
            return (255, 255, 255)

        # Detect using Mean strategy (fast and accurate)
        bg_color = detect_color_mean(border_pixels)

        # Validate color (check for NaN)
        if np.any(np.isnan(bg_color)):
            print(f"  [WARNING] Invalid color detected for bbox: {bbox}, using white background")
            return (255, 255, 255)

        return bg_color

    except Exception as e:
        print(f"  [ERROR] Color detection failed for bbox {bbox}: {e}, using white background")
        return (255, 255, 255)


def replace_letters_preserve_formulas(text: str) -> str:
    """
    Replace all letters with 'c' while preserving LaTeX formulas and spaces.

    Args:
        text: Input text with possible LaTeX formulas

    Returns:
        Text with letters replaced

    Examples:
        "Hello world" -> "ccccc ccccc"
        "Einstein's $E=mc^2$ theory" -> "ccccccccc'c $E=mc^2$ cccccc"
        "Test 123" -> "cccc 123"
    """
    # Extract LaTeX formulas and replace with placeholders
    # Use placeholders without letters to avoid being replaced
    formula_pattern = r'\$(.+?)\$'
    formulas = []

    def extract_formula(match):
        formulas.append(match.group(0))  # Store with $ delimiters
        return f"###_{len(formulas)-1}_###"  # Use ### instead of letters

    # Replace formulas with placeholders
    text_without_formulas = re.sub(formula_pattern, extract_formula, text)

    # Replace all letters (a-z, A-Z) with 'c', but preserve spaces and other characters
    replaced_text = re.sub(r'[a-zA-Z]', 'c', text_without_formulas)

    # Restore formulas
    for i, formula in enumerate(formulas):
        replaced_text = replaced_text.replace(f"###_{i}_###", formula)

    return replaced_text


# Reuse functions from demo_pdf_render_enhanced.py

def parse_mixed_content(text: str) -> list:
    """
    Parse text and extract LaTeX formulas.

    Args:
        text: Input text with LaTeX formulas in $...$ format

    Returns:
        List of (type, content) tuples where type is 'text' or 'latex'
    """
    segments = []
    pattern = r'\$(.+?)\$'
    last_end = 0

    for match in re.finditer(pattern, text):
        # Add text before formula
        if match.start() > last_end:
            text_segment = text[last_end:match.start()]
            if text_segment:
                segments.append(('text', text_segment))

        # Add formula (without $ delimiters)
        segments.append(('latex', match.group(1)))
        last_end = match.end()

    # Add remaining text after last formula
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            segments.append(('text', remaining))

    # If no formulas found, return the whole text as one segment
    if not segments:
        segments.append(('text', text))

    return segments


def render_latex_to_bytes_with_size(latex_formula: str, fontsize: int = 20, dpi: int = 300) -> tuple:
    """
    Render a LaTeX formula to PNG image bytes and return size info.
    Automatically crops transparent borders to minimize white space.

    Args:
        latex_formula: LaTeX formula string (without $ delimiters)
        fontsize: Font size for rendering
        dpi: DPI for image resolution

    Returns:
        Tuple of (image_bytes, width_pixels, height_pixels)
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)

    # Render the LaTeX formula
    text = fig.text(0, 0, f'${latex_formula}$', fontsize=fontsize)

    # Get the bounding box
    fig.canvas.draw()
    bbox = text.get_window_extent(fig.canvas.get_renderer())

    # Adjust figure size to fit the text
    fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)

    # Re-position text
    text.set_position((0, 0))

    # Save to bytes buffer with minimal padding
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                transparent=True, pad_inches=0.02)
    plt.close(fig)

    buf.seek(0)

    # Load image and crop transparent borders
    pil_img = Image.open(buf)

    # Get the bounding box of non-transparent pixels
    if pil_img.mode == 'RGBA':
        alpha = pil_img.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            margin = 2
            bbox = (
                max(0, bbox[0] - margin),
                max(0, bbox[1] - margin),
                min(pil_img.width, bbox[2] + margin),
                min(pil_img.height, bbox[3] + margin)
            )
            pil_img = pil_img.crop(bbox)

    # Convert back to bytes
    output_buf = io.BytesIO()
    pil_img.save(output_buf, format='PNG')
    output_buf.seek(0)
    img_bytes = output_buf.getvalue()

    width, height = pil_img.size

    return img_bytes, width, height


def calculate_text_dimensions(text: str, fontname: str, fontsize: float, max_width: float, line_spacing: float = 1.2) -> tuple:
    """
    Calculate dimensions of text when rendered with wrapping.
    Simulates the actual line-breaking logic used in render_mixed_content.

    Height calculation matches render_mixed_content:
    - First line: fontsize (baseline at 0.85, descender at 0.15)
    - Subsequent lines: (n-1) * fontsize * line_spacing
    - Total: fontsize + (n-1) * fontsize * line_spacing

    Args:
        text: Text to measure
        fontname: Font name
        fontsize: Font size
        max_width: Maximum width before wrapping
        line_spacing: Line spacing multiplier (default 1.2)

    Returns:
        Tuple of (total_width, total_height, num_lines)
    """
    if not text:
        return 0, 0, 0

    # Check if text contains CJK characters
    # This determines which font will be used for rendering
    has_cjk = any(is_cjk_char(ch) for ch in text)

    # Character width function matching render_mixed_content logic
    # Use conservative estimates to prevent right overflow
    def get_char_width(ch):
        if is_cjk_char(ch):
            # Full-width characters (CJK ideographs, punctuation, etc.)
            return fontsize * 1.0
        elif has_cjk:
            # When using CJK font for mixed text:
            # Use more conservative (larger) estimates to prevent overflow
            if ch == ' ':
                return fontsize * 0.4  # Space
            elif ch in 'mwMW@':
                return fontsize * 0.8  # Wide letters
            elif ch in 'il1!|.,;:\'"':
                return fontsize * 0.35  # Narrow characters
            elif ch in 'fjrt':
                return fontsize * 0.45  # Slightly narrow
            else:
                return fontsize * 0.6  # Average - be conservative
        else:
            # Pure Latin text using helv font
            return fitz.get_text_length(ch, fontname=fontname, fontsize=fontsize)

    # Simulate line-by-line rendering
    cursor_x = 0
    num_lines = 1
    max_line_width = 0

    for ch in text:
        # Handle newline
        if ch == '\n':
            max_line_width = max(max_line_width, cursor_x)
            cursor_x = 0
            num_lines += 1
            continue

        # Calculate character width
        char_width = get_char_width(ch)

        # Check if need to wrap
        if cursor_x + char_width > max_width:
            max_line_width = max(max_line_width, cursor_x)
            cursor_x = 0
            num_lines += 1
            # Skip leading space on new line
            if ch == ' ':
                continue

        cursor_x += char_width

    max_line_width = max(max_line_width, cursor_x)

    # Height calculation matching render_mixed_content:
    # First line takes fontsize, subsequent lines add fontsize * line_spacing each
    # total = fontsize + (num_lines - 1) * fontsize * line_spacing
    if num_lines == 1:
        total_height = fontsize
    else:
        total_height = fontsize + (num_lines - 1) * fontsize * line_spacing

    return max_line_width, total_height, num_lines


def calculate_adaptive_font_size(text: str, bbox: fitz.Rect,
                                   default_size: int = 10, min_size: int = 4,
                                   fontname: str = "helv") -> tuple:
    """
    Calculate adaptive font size and line spacing.

    Strategy (user preference: don't enlarge font, keep readable line spacing):
    1. Start with default font size and normal line spacing (1.2)
    2. If doesn't fit, SHRINK font size first (keep line spacing)
    3. Only reduce line spacing as last resort if min font still doesn't fit

    Args:
        text: Text to fit in bbox
        bbox: Bounding rectangle
        default_size: Default/maximum font size (default: 10pt)
        min_size: Minimum size threshold (default: 4pt)

    Returns:
        Tuple of (font_size, line_spacing) that fits the bbox
    """
    bbox_width = bbox.x1 - bbox.x0
    bbox_height = bbox.y1 - bbox.y0

    # Match the margin values in render_mixed_content
    margin_x = 2  # Must match render_mixed_content
    margin_y = 1

    # Calculate actual usable space
    usable_width = bbox_width - margin_x * 2
    usable_height = bbox_height - margin_y * 2 - 2  # 2pt safety margin for height

    # Default line spacing - keep it readable
    default_line_spacing = 1.2

    # Step 1: Try default size with normal line spacing
    _, height, _ = calculate_text_dimensions(text, fontname, default_size, usable_width, default_line_spacing)
    if height <= usable_height:
        return default_size, default_line_spacing

    # Step 2: Shrink font size (keep normal line spacing)
    low, high = min_size, default_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2
        _, height, _ = calculate_text_dimensions(text, fontname, mid, usable_width, default_line_spacing)

        if height <= usable_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    # Check if we found a working size with normal line spacing
    _, height, _ = calculate_text_dimensions(text, fontname, best_size, usable_width, default_line_spacing)
    if height <= usable_height:
        return best_size, default_line_spacing

    # Step 3: Last resort - use minimum font and reduce line spacing
    line_spacings = [1.1, 1.0, 0.95, 0.9]
    for line_spacing in line_spacings:
        _, height, _ = calculate_text_dimensions(text, fontname, min_size, usable_width, line_spacing)
        if height <= usable_height:
            return min_size, line_spacing

    # Absolute fallback
    return min_size, 0.9


# ==================== 两阶段字号分析（方案 E） ====================

# Category 字号范围配置
CATEGORY_FONTSIZE_CONFIG = {
    'title':          {'default': 16, 'min': 10, 'max': 24},
    'section-header': {'default': 13, 'min': 9,  'max': 18},
    'text':           {'default': 10, 'min': 5,  'max': 14},
    'list-item':      {'default': 10, 'min': 5,  'max': 14},
    'caption':        {'default': 9,  'min': 5,  'max': 12},
    'footnote':       {'default': 8,  'min': 4,  'max': 10},
    'page-header':    {'default': 8,  'min': 4,  'max': 10},
    'page-footer':    {'default': 8,  'min': 4,  'max': 10},
}


def estimate_fontsize_from_bbox(bbox_height: float, text: str, scale_factor: float) -> float:
    """
    根据 bbox 高度和文本估算原始字号。

    Args:
        bbox_height: bbox 的高度（PDF 坐标，已缩放）
        text: 文本内容
        scale_factor: 坐标缩放比例（PDF DPI / 图片 DPI）

    Returns:
        估算的字号（pt）
    """
    # 估算行数：根据换行符数量 + 1
    line_count = text.count('\n') + 1

    # 如果文本很长但没有换行符，可能是需要自动换行的
    # 这种情况下行数估算不准确，返回 0 表示无法准确估算
    text_length = len(text.replace('\n', '').replace(' ', ''))
    if line_count == 1 and text_length > 50:
        return 0  # 无法准确估算，需要用默认值

    # 字号估算：bbox 高度 / 行数 / 行高系数
    # 行高系数通常在 1.2-1.5 之间，这里使用 1.3
    line_height_ratio = 1.3
    estimated_fontsize = bbox_height / line_count / line_height_ratio

    return estimated_fontsize


def analyze_category_fontsizes(pages_data: list, scale_factor: float) -> dict:
    """
    第一阶段：扫描所有元素，分析各 category 的典型字号。

    Args:
        pages_data: 页面数据列表
        scale_factor: 坐标缩放比例

    Returns:
        各 category 的推荐字号字典 {category: fontsize}
    """
    # 收集各 category 的字号样本
    category_samples = {}

    for page_data in pages_data:
        elements = page_data.get('elements', [])

        for element in elements:
            bbox = element.get('bbox')
            text = element.get('text', '')
            category = element.get('category', 'text').lower()

            # 跳过特殊类型
            if category in ['table', 'formula', 'picture']:
                continue

            if not bbox or not text:
                continue

            # 计算 bbox 高度（转换到 PDF 坐标）
            x0, y0, x1, y1 = bbox
            bbox_height = (y1 - y0) * scale_factor

            # 估算字号
            estimated_size = estimate_fontsize_from_bbox(bbox_height, text, scale_factor)

            if estimated_size > 0:
                if category not in category_samples:
                    category_samples[category] = []
                category_samples[category].append({
                    'estimated_size': estimated_size,
                    'bbox_height': bbox_height,
                    'text_length': len(text),
                    'line_count': text.count('\n') + 1,
                })

    # 计算各 category 的典型字号
    category_fontsizes = {}

    for category, samples in category_samples.items():
        if not samples:
            continue

        # 获取配置的字号范围
        config = CATEGORY_FONTSIZE_CONFIG.get(category, {
            'default': 10, 'min': 5, 'max': 14
        })

        # 策略：使用中位数（更稳定，不受极端值影响）
        # 只考虑单行或少行的样本（估算更准确）
        reliable_samples = [s for s in samples if s['line_count'] <= 3]

        if reliable_samples:
            sizes = [s['estimated_size'] for s in reliable_samples]
            sizes.sort()
            median_size = sizes[len(sizes) // 2]

            # 限制在配置范围内
            final_size = max(config['min'], min(config['max'], median_size))
        else:
            # 没有可靠样本，使用默认值
            final_size = config['default']

        category_fontsizes[category] = round(final_size, 1)

    # 确保层次关系：title > section-header > text > caption/footnote
    # 如果分析结果违反层次关系，进行调整
    category_fontsizes = enforce_fontsize_hierarchy(category_fontsizes)

    return category_fontsizes


def enforce_fontsize_hierarchy(fontsizes: dict) -> dict:
    """
    确保字号满足层次关系：title > section-header > text > caption/footnote

    Args:
        fontsizes: 各 category 的字号字典

    Returns:
        调整后的字号字典
    """
    result = fontsizes.copy()

    # 定义层次关系（从大到小）
    hierarchy = [
        ('title', 16),
        ('section-header', 13),
        ('text', 10),
        ('list-item', 10),
        ('caption', 9),
        ('footnote', 8),
        ('page-header', 8),
        ('page-footer', 8),
    ]

    # 获取基准字号（text 的字号，或默认 10）
    base_size = result.get('text', 10)

    # 从上往下调整：确保上层 >= 下层
    prev_size = 999
    for category, default_size in hierarchy:
        if category in result:
            current_size = result[category]
            # 确保当前 <= 上一层
            if current_size > prev_size:
                current_size = prev_size
            result[category] = current_size
            prev_size = current_size
        else:
            # 如果没有该 category，使用默认值（但不超过上一层）
            result[category] = min(default_size, prev_size)
            prev_size = result[category]

    # 从下往上调整：确保上层 > 下层（至少差 1pt）
    prev_size = 0
    for category, default_size in reversed(hierarchy):
        if category in result:
            current_size = result[category]
            # 确保当前 > 下一层（至少差 0.5pt）
            if current_size <= prev_size:
                current_size = prev_size + 0.5
            result[category] = round(current_size, 1)
            prev_size = current_size

    return result


def get_fontsize_for_element(category: str, bbox: fitz.Rect, text: str,
                              category_fontsizes: dict) -> tuple:
    """
    获取元素的字号（结合分析结果和自适应缩放）。

    Args:
        category: 元素类别
        bbox: 元素边界框
        text: 元素文本
        category_fontsizes: 第一阶段分析得到的各 category 字号

    Returns:
        (fontsize, line_spacing) 元组
    """
    # 获取该 category 的推荐字号
    config = CATEGORY_FONTSIZE_CONFIG.get(category, {'default': 10, 'min': 5, 'max': 14})
    recommended_size = category_fontsizes.get(category, config['default'])
    min_size = config['min']

    # 计算文本是否能放下
    text_only = re.sub(r'\$.*?\$', 'FORMULA', text)

    # 尝试使用推荐字号
    _, height, _ = calculate_text_dimensions(text_only, "helv", recommended_size,
                                              bbox.width - 4, 1.2)

    usable_height = bbox.height - 4  # 减去边距

    if height <= usable_height:
        # 推荐字号能放下
        return recommended_size, 1.2

    # 推荐字号放不下，需要缩小
    # 使用二分查找找到合适的字号
    low, high = min_size, recommended_size
    best_size = min_size

    while low <= high:
        mid = (low + high) / 2
        _, height, _ = calculate_text_dimensions(text_only, "helv", mid,
                                                  bbox.width - 4, 1.2)
        if height <= usable_height:
            best_size = mid
            low = mid + 0.5
        else:
            high = mid - 0.5

    # 如果最小字号还是放不下，尝试减小行距
    _, height, _ = calculate_text_dimensions(text_only, "helv", best_size,
                                              bbox.width - 4, 1.2)

    if height <= usable_height:
        return round(best_size, 1), 1.2

    # 尝试减小行距
    for line_spacing in [1.1, 1.0, 0.95, 0.9]:
        _, height, _ = calculate_text_dimensions(text_only, "helv", min_size,
                                                  bbox.width - 4, line_spacing)
        if height <= usable_height:
            return min_size, line_spacing

    return min_size, 0.9


def render_mixed_content(page: fitz.Page, text: str, bbox: fitz.Rect,
                          fontsize: int, fontname: str = "helv", bg_color_rgb: tuple = (255, 255, 255),
                          line_spacing: float = None):
    """
    Render text with inline LaTeX formulas using character-by-character processing.

    Args:
        page: PDF page object
        text: Text with embedded LaTeX formulas
        bbox: Bounding rectangle
        fontsize: Font size to use
        fontname: Font name
        bg_color_rgb: Background color RGB tuple (r, g, b), range 0-255
        line_spacing: Line spacing multiplier (if None, auto-detect from language)
    """
    segments = parse_mixed_content(text)

    # Use provided line_spacing or auto-detect from language
    if line_spacing is None:
        lang = detect_language(text)
        line_spacing = LANG_LINEHEIGHT_MAP.get(lang, 1.2)

    # Auto-detect text color based on background brightness
    brightness = (bg_color_rgb[0] * 299 + bg_color_rgb[1] * 587 + bg_color_rgb[2] * 114) / 1000
    if brightness > 128:
        text_color = (0, 0, 0)  # Black text on light background
    else:
        text_color = (1, 1, 1)  # White text on dark background

    # Check if text contains CJK characters and load appropriate font
    has_cjk = any(is_cjk_char(ch) for ch in text)
    cjk_fontname = None
    cjk_fontfile = None

    if has_cjk:
        # Try to find a CJK font
        import os
        cjk_font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
            "C:/Windows/Fonts/simsun.ttc",    # SimSun
            "C:/Windows/Fonts/simhei.ttf",    # SimHei
        ]
        for font_path in cjk_font_paths:
            if os.path.exists(font_path):
                cjk_fontfile = font_path
                cjk_fontname = "cjk-font"
                break

    # Margins - be conservative to prevent overflow
    margin_x = 2  # Horizontal margin (increased for safety)
    margin_y = 1  # Vertical margin

    cursor_x = bbox.x0 + margin_x
    cursor_y = bbox.y0 + margin_y + fontsize * 0.85  # Adjust baseline position
    line_height = fontsize * line_spacing
    max_x = bbox.x1 - margin_x  # Right boundary

    # Character buffer for batch insertion
    cstk = ""  # Character stack
    cstk_x = cursor_x  # Starting x position of buffer

    def flush_buffer():
        """Output the character buffer to PDF using appropriate font."""
        nonlocal cstk, cursor_x, cstk_x
        if not cstk:
            return

        # When we have CJK content, use CJK font for ALL characters
        # This ensures consistent appearance for mixed text
        if has_cjk and cjk_fontfile:
            page.insert_text(
                point=(cstk_x, cursor_y),
                text=cstk,
                fontsize=fontsize,
                fontname=cjk_fontname,
                fontfile=cjk_fontfile,
                color=text_color
            )
        else:
            page.insert_text(
                point=(cstk_x, cursor_y),
                text=cstk,
                fontsize=fontsize,
                fontname=fontname,
                color=text_color
            )

        cstk = ""

    # Character width calculation function
    # When using CJK font, all characters are rendered with that font
    # Use conservative estimates to prevent right overflow
    def get_char_width(ch):
        if is_cjk_char(ch):
            # Full-width characters (CJK ideographs, punctuation, etc.)
            return fontsize * 1.0
        elif has_cjk and cjk_fontfile:
            # When using CJK font for mixed text:
            # Use more conservative (larger) estimates to prevent overflow
            if ch == ' ':
                return fontsize * 0.4  # Space
            elif ch in 'mwMW@':
                return fontsize * 0.8  # Wide letters
            elif ch in 'il1!|.,;:\'"':
                return fontsize * 0.35  # Narrow characters
            elif ch in 'fjrt':
                return fontsize * 0.45  # Slightly narrow
            else:
                return fontsize * 0.6  # Average - be conservative
        else:
            # Pure Latin text using helv font
            return fitz.get_text_length(ch, fontname=fontname, fontsize=fontsize)

    for seg_type, content in segments:
        if seg_type == 'text':
            # Process character by character
            for i, ch in enumerate(content):
                # Handle newline character
                if ch == '\n':
                    flush_buffer()
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height
                    cstk_x = cursor_x
                    continue
                # Calculate character width
                char_width = get_char_width(ch)

                # Check if character fits on current line
                if cursor_x + char_width > max_x:
                    # Flush buffer before wrapping
                    flush_buffer()

                    # Move to next line
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height

                    # Skip leading space on new line
                    if ch == ' ':
                        continue
                if not cstk:
                    cstk_x = cursor_x
                # Add character to buffer
                cstk += ch
                cursor_x += char_width

            # Flush remaining characters
            flush_buffer()

        elif seg_type == 'latex':
            # Flush text buffer before inserting formula
            flush_buffer()

            try:
                # Render formula with larger fontsize for better quality
                formula_fontsize = fontsize * 0.9  # Increase formula size
                formula_bytes, img_width_px, img_height_px = render_latex_to_bytes_with_size(
                    content, fontsize=formula_fontsize, dpi=300
                )

                # Convert pixels to points: 72 points/inch, 300 dpi
                # Formula: points = pixels * (72 / dpi)
                px_to_pt = 72.0 / 300.0  # 0.24
                img_width_pt = img_width_px * px_to_pt
                img_height_pt = img_height_px * px_to_pt

                # Check if formula fits on current line
                if cursor_x + img_width_pt > max_x:
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height
                    cstk_x = cursor_x

                img_rect = fitz.Rect(
                    cursor_x,
                    cursor_y - img_height_pt * 0.90,
                    cursor_x + img_width_pt,
                    cursor_y + img_height_pt * 0.10
                )

                page.insert_image(img_rect, stream=formula_bytes)
                cursor_x += img_width_pt + 2  # Small spacing after formula
                cstk_x = cursor_x

            except Exception as e:
                print(f"Warning: Failed to render formula '{content}': {e}")
                fallback_text = f"${content}$"
                fallback_width = fitz.get_text_length(fallback_text, fontname=fontname, fontsize=fontsize)

                if cursor_x + fallback_width > max_x:
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height
                    cstk_x = cursor_x

                page.insert_text(
                    point=(cursor_x, cursor_y),
                    text=fallback_text,
                    fontsize=fontsize,
                    fontname=fontname,
                    color=(1, 0, 0)
                )
                cursor_x += fallback_width
                cstk_x = cursor_x

    # Final flush in case there's remaining buffer
    flush_buffer()


def _render_table_to_pdf(table_html, width, height, fontsize, line_height, padding, bg_color, text_color, border_color):
    """
    内部函数：使用WeasyPrint渲染HTML表格为PDF并返回文档和内容边界
    """
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: {width}pt {height * 3}pt;
                margin: 0;
            }}
            body {{
                margin: 0;
                padding: 0;
                font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
            }}
            table {{
                width: {width}pt;
                max-width: {width}pt;
                border-collapse: collapse;
                font-size: {fontsize}pt;
                table-layout: fixed;
                background-color: {bg_color};
                color: {text_color};
            }}
            th, td {{
                border: 0.5pt solid {border_color};
                padding: {padding}pt;
                text-align: left;
                vertical-align: middle;
                word-wrap: break-word;
                word-break: break-word;
                overflow-wrap: break-word;
                overflow: hidden;
                line-height: {line_height};
            }}
            th {{
                font-weight: bold;
            }}
            td:nth-child(n+3) {{
                text-align: right;
            }}
            strong {{
                font-weight: bold;
                color: {text_color};
            }}
        </style>
    </head>
    <body>{table_html}</body>
    </html>
    """

    pdf_bytes = BytesIO()
    HTML(string=full_html).write_pdf(pdf_bytes)
    pdf_bytes.seek(0)

    temp_doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    temp_page = temp_doc[0]

    # 获取表格的实际边界
    content_rects = []
    drawings = temp_page.get_drawings()
    for d in drawings:
        content_rects.append(d["rect"])

    text_dict = temp_page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if "bbox" in block:
            content_rects.append(fitz.Rect(block["bbox"]))

    if content_rects:
        min_x = min(r.x0 for r in content_rects)
        min_y = min(r.y0 for r in content_rects)
        max_x = max(r.x1 for r in content_rects)
        max_y = max(r.y1 for r in content_rects)
        content_rect = fitz.Rect(min_x, min_y, max_x, max_y)
    else:
        content_rect = temp_page.rect

    return temp_doc, content_rect


def render_table(page, bbox, table_html, bg_color_rgb=(255, 255, 255)):
    """
    使用WeasyPrint渲染HTML表格到PDF页面的指定区域

    高效自适应算法：只渲染2次，根据首次渲染结果直接计算最优字号。

    Args:
        page: fitz.Page对象
        bbox: Tuple of (x0, y0, x1, y1) 目标区域坐标
        table_html: HTML表格字符串
        bg_color_rgb: 背景颜色RGB元组 (r, g, b)，范围0-255
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    start_time = time.time()

    # 根据背景色自动选择文本和边框颜色
    # 计算背景亮度
    brightness = (bg_color_rgb[0] * 299 + bg_color_rgb[1] * 587 + bg_color_rgb[2] * 114) / 1000
    if brightness > 128:
        # 浅色背景：黑色文本和边框
        text_color = "#000000"
        border_color = "#000000"
    else:
        # 深色背景：白色文本和边框
        text_color = "#ffffff"
        border_color = "#ffffff"

    bg_color = f"#{bg_color_rgb[0]:02x}{bg_color_rgb[1]:02x}{bg_color_rgb[2]:02x}"

    # 初始参数
    init_fontsize = 10.0
    line_height = 1.2
    padding = 2.0

    # 第1次渲染：获取初始大小
    temp_doc, content_rect = _render_table_to_pdf(
        table_html, width, height, init_fontsize, line_height, padding,
        bg_color, text_color, border_color
    )

    scale_x = width / content_rect.width if content_rect.width > 0 else 1
    scale_y = height / content_rect.height if content_rect.height > 0 else 1
    scale = min(scale_x, scale_y, 1.0)

    # 如果表格刚好能放入bbox（scale在0.90-1.0之间），直接使用
    if 0.90 <= scale <= 1.0:
        final_doc = temp_doc
        final_rect = content_rect
    else:
        temp_doc.close()

        # 计算最优字号，让表格刚好填满bbox
        # 字号与表格高度关系近似平方根（不是线性）
        optimal_fontsize = init_fontsize * math.sqrt(scale) * 0.95
        optimal_line_height = max(1.0, 1.0 + (line_height - 1.0) * math.sqrt(scale))
        optimal_padding = max(0.5, padding * math.sqrt(scale))

        # 第2次渲染：使用优化后的参数
        final_doc, final_rect = _render_table_to_pdf(
            table_html, width, height, optimal_fontsize, optimal_line_height, optimal_padding,
            bg_color, text_color, border_color
        )

        scale_x = width / final_rect.width if final_rect.width > 0 else 1
        scale_y = height / final_rect.height if final_rect.height > 0 else 1
        scale = min(scale_x, scale_y, 1.0)

    # 计算最终渲染尺寸
    actual_width = final_rect.width * scale
    actual_height = final_rect.height * scale
    target_rect = fitz.Rect(x0, y0, x0 + actual_width, y0 + actual_height)

    # 将渲染结果嵌入到目标页面
    page.show_pdf_page(target_rect, final_doc, 0, clip=final_rect)

    final_doc.close()

    elapsed_time = time.time() - start_time
    print(f"    [OK] 表格渲染完成 (尺寸: {actual_width:.1f}x{actual_height:.1f}pt, 耗时: {elapsed_time:.2f}s)")


def process_pdf_with_json(pdf_path: str, json_path: str = None, output_path: str = None, image_dpi: int = 200, draw_bbox: bool = False,
                          translate: bool = False, src_lang: str = "en", tgt_lang: str = "zh",
                          model_type: str = None, app_id: str = None):
    """
    Process PDF with OCR JSON: optionally translate and render back.

    Args:
        pdf_path: Input PDF file path
        json_path: OCR JSON file path (optional, if not provided, will call OCR API)
        output_path: Output PDF path (optional)
        image_dpi: DPI of the image used for OCR (default: 200)
        draw_bbox: Whether to draw bbox borders for debugging (default: False)
        translate: Whether to translate text elements (default: False)
        src_lang: Source language for translation (default: "en")
        tgt_lang: Target language for translation (default: "zh")
        model_type: Translation model type (default: None, uses deepseek_v3)
        app_id: Application ID for translation API (default: None)
    """
    print("\n" + "=" * 80)
    if translate:
        print(f"PDF文本覆盖处理 - 翻译模式 ({src_lang} -> {tgt_lang})")
    else:
        print("PDF文本覆盖处理")
    print("=" * 80)

    # Validate input files
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"[ERROR] PDF文件不存在: {pdf_path}")
        return

    # Load OCR data from JSON file or call OCR API
    if json_path is not None:
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"[ERROR] JSON文件不存在: {json_path}")
            return

        # Load JSON
        print(f"\n[1/5] 加载JSON文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    else:
        # Call OCR API to extract PDF content
        print(f"\n[1/5] 调用OCR API提取PDF内容...")
        ocr_data = extract_pdf_with_ocr(pdf_path, image_dpi=image_dpi)

    # Open PDF
    print(f"[2/5] 打开PDF文件: {pdf_path}")
    doc = fitz.open(pdf_path)

    # Determine output path
    if output_path is None:
        output_path = f"output/{pdf_file.stem}_overlay.pdf"

    # Support two JSON formats:
    # Format 1: {"pages": [{"page_index": 0, "elements": [...]}]}
    # Format 2: {"page_index": 0, "elements": [...]}
    if 'pages' in ocr_data:
        pages_data = ocr_data['pages']
    elif 'page_index' in ocr_data and 'elements' in ocr_data:
        # Single page format, wrap it in a list
        pages_data = [ocr_data]
    else:
        print(f"[ERROR] 无法识别的JSON格式！")
        print(f"  支持的格式1: {{'pages': [{{'page_index': 0, 'elements': [...]}}]}}")
        print(f"  支持的格式2: {{'page_index': 0, 'elements': [...]}}")
        return

    print(f"[3/6] 处理 {len(pages_data)} 个页面...")
    print(f"  坐标转换: 图片DPI {image_dpi} -> PDF DPI 72")

    # Calculate scaling factor
    pdf_dpi = 72.0
    scale_factor = pdf_dpi / image_dpi
    print(f"  缩放比例: {scale_factor:.4f}")

    # ========== 第一阶段：分析各 category 的典型字号 ==========
    print(f"\n[4/6] 分析字号...")
    category_fontsizes = analyze_category_fontsizes(pages_data, scale_factor)
    print(f"  分析完成，各类别推荐字号:")
    for cat, size in sorted(category_fontsizes.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {size}pt")

    # ========== 第二阶段：渲染 ==========
    print(f"\n[5/6] 渲染页面...")

    # Process each page
    total_elements = 0
    for page_data in pages_data:
        page_index = page_data.get('page_index', 0)
        page = doc[page_index]

        elements = page_data.get('elements', [])
        print(f"\n  页面 {page_index}: {len(elements)} 个元素")

        for elem_idx, element in enumerate(elements):
            bbox = element.get('bbox')
            text = element.get('text', '')
            category = element.get('category', 'text').lower()  # Normalize to lowercase

            if not bbox:
                continue

            # For table elements, allow empty text (will contain HTML)
            if category != 'table' and not text:
                continue

            # Convert bbox to tuple
            if isinstance(bbox, list):
                bbox = tuple(bbox)

            # Convert image coordinates to PDF coordinates (simple scaling)
            x0, y0, x1, y1 = bbox
            bbox = (
                x0 * scale_factor,
                y0 * scale_factor,
                x1 * scale_factor,
                y1 * scale_factor
            )
            # Fix: Move entire text box up by 2 pixels to align with original PDF position
            bbox = (bbox[0], bbox[1] - 2, bbox[2], bbox[3] - 2)

            # Detect background color
            bg_color = detect_background_color(page, bbox)
            bg_color_norm = tuple(c / 255 for c in bg_color)

            # Draw background rectangle (slightly expanded to cover edge artifacts)
            rect = fitz.Rect(bbox)
            expand = 2  # 扩展像素数，防止漏出底部文本
            expanded_rect = fitz.Rect(
                rect.x0 - expand,
                rect.y0 - expand,
                rect.x1 + expand,
                rect.y1 + expand
            )
            page.draw_rect(expanded_rect, color=bg_color_norm, fill=bg_color_norm)

            # ========== Category-based Rendering ==========
            if category == 'table':
                # Table rendering path (WeasyPrint)
                try:
                    print(f"    [表格] 元素 {elem_idx + 1}")
                    render_table(page, bbox, text, bg_color_rgb=bg_color)
                except Exception as e:
                    print(f"    [错误] 表格渲染失败: {e}")
                    page.insert_textbox(rect, "错误: 表格渲染失败",
                                       fontsize=10, color=(1, 0, 0))
            elif category == 'formula':
                # Formula rendering path (display formula)
                try:
                    print(f"    [公式] 元素 {elem_idx + 1}")

                    # Extract formula content (remove $$ delimiters)
                    formula_text = text.strip()
                    if formula_text.startswith('$$') and formula_text.endswith('$$'):
                        formula_content = formula_text[2:-2].strip()
                    elif formula_text.startswith('$') and formula_text.endswith('$'):
                        formula_content = formula_text[1:-1].strip()
                    else:
                        formula_content = formula_text

                    # Calculate appropriate font size based on bbox height
                    bbox_height = rect.height
                    # Use larger font size for display formulas
                    formula_fontsize = min(bbox_height * 0.6, 24)  # Max 24pt
                    formula_fontsize = max(formula_fontsize, 14)   # Min 14pt

                    # Render formula to image
                    formula_bytes, img_width_px, img_height_px = render_latex_to_bytes_with_size(
                        formula_content, fontsize=formula_fontsize, dpi=300
                    )

                    # Convert pixels to points
                    px_to_pt = 72.0 / 300.0
                    img_width_pt = img_width_px * px_to_pt
                    img_height_pt = img_height_px * px_to_pt

                    # Calculate scaling to fit in bbox while maintaining aspect ratio
                    scale_x = rect.width / img_width_pt if img_width_pt > 0 else 1
                    scale_y = rect.height / img_height_pt if img_height_pt > 0 else 1
                    scale = min(scale_x, scale_y, 1.0)  # Don't upscale

                    final_width = img_width_pt * scale
                    final_height = img_height_pt * scale

                    # Center the formula in the bbox
                    center_x = rect.x0 + (rect.width - final_width) / 2
                    center_y = rect.y0 + (rect.height - final_height) / 2

                    # Create image rectangle
                    img_rect = fitz.Rect(
                        center_x,
                        center_y,
                        center_x + final_width,
                        center_y + final_height
                    )

                    # Insert formula image
                    page.insert_image(img_rect, stream=formula_bytes)
                    print(f"    [OK] 公式渲染完成 (尺寸: {final_width:.1f}x{final_height:.1f}pt)")

                except Exception as e:
                    print(f"    [错误] 公式渲染失败: {e}")
                    page.insert_textbox(rect, f"错误: 公式渲染失败\n{text[:50]}",
                                       fontsize=8, color=(1, 0, 0))
            else:
                # Text/Title/Caption rendering path
                modified_text = text

                # Translate if enabled
                if translate and not should_skip_translation_check(text, category):
                    translated_text = translate_element_text(
                        text, category,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        model_type=model_type,
                        app_id=app_id,
                    )
                    if translated_text != text:
                        modified_text = translated_text
                        print(f"    [翻译] 元素 {elem_idx + 1}: OK")
                        print(f"      原文: {text[:80]}{'...' if len(text) > 80 else ''}")
                        print(f"      译文: {translated_text[:80]}{'...' if len(translated_text) > 80 else ''}")

                # 使用两阶段分析得到的字号（方案 E）
                fontsize, line_spacing = get_fontsize_for_element(
                    category, rect, modified_text, category_fontsizes
                )

                # Print font size for each text element (handle encoding for Windows console)
                display_text = text[:40].replace('\n', ' ')
                if len(text) > 40:
                    display_text += "..."
                # Encode then decode to handle characters that can't be displayed
                try:
                    print(f"    [{category}] elem {elem_idx + 1}: {fontsize}pt | {display_text}")
                except UnicodeEncodeError:
                    safe_text = display_text.encode('ascii', errors='replace').decode('ascii')
                    print(f"    [{category}] elem {elem_idx + 1}: {fontsize}pt | {safe_text}")

                # Render mixed content with auto color adaptation and calculated line spacing
                render_mixed_content(page, modified_text, rect, fontsize, bg_color_rgb=bg_color, line_spacing=line_spacing)

            # Draw bbox border if requested
            if draw_bbox:
                # Draw border in red color for visibility
                page.draw_rect(rect, color=(1, 0, 0), width=1.5)

            total_elements += 1

            if (elem_idx + 1) % 10 == 0:
                print(f"    处理进度: {elem_idx + 1}/{len(elements)}")

    # Save output
    print(f"\n[6/6] 保存输出PDF: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    print(f"完成!")
    print("\n" + "=" * 80)
    print("处理统计:")
    print(f"  - 总页数: {len(pages_data)}")
    print(f"  - 总元素: {total_elements}")
    print(f"  - 输出文件: {output_path}")
    print("=" * 80 + "\n")


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='PDF文本覆盖工具 - 使用OCR提取PDF内容并渲染回PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接处理PDF（自动调用OCR API）
  python demo_text_overlay.py input.pdf

  # 使用已有的JSON文件
  python demo_text_overlay.py input.pdf -j ocr_result.json

  # 指定输出路径
  python demo_text_overlay.py input.pdf -o custom_output.pdf

  # 启用翻译
  python demo_text_overlay.py input.pdf --translate --src-lang en --tgt-lang zh

JSON格式:
  {
    "pages": [
      {
        "page_index": 0,
        "elements": [
          {
            "bbox": [x0, y0, x1, y1],
            "category": "text",
            "text": "Original text with $E=mc^2$ formula"
          }
        ]
      }
    ]
  }
        """
    )

    parser.add_argument('pdf_path', type=str, help='输入PDF文件路径')
    parser.add_argument('-j', '--json', type=str, default=None, dest='json_path',
                       help='OCR JSON文件路径（可选，不提供则自动调用OCR API）')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出PDF文件路径 (默认: output/{input}_overlay.pdf)')
    parser.add_argument('--dpi', type=int, default=200,
                       help='OCR图片的DPI (默认: 200)')
    parser.add_argument('--draw-bbox', action='store_true',
                       help='在PDF上绘制bbox边框（调试用）')

    # Translation options
    parser.add_argument('--translate', action='store_true',
                       help='启用翻译功能')
    parser.add_argument('--src-lang', type=str, default='en',
                       help='源语言 (默认: zh)')
    parser.add_argument('--tgt-lang', type=str, default='zh',
                       help='目标语言 (默认: en)')
    parser.add_argument('--model', type=str, default=None,
                       help='翻译模型 (deepseek_v3, volcengine)')
    parser.add_argument('--app-id', type=str, default=None,
                       help='OCR/翻译API的应用ID (默认: test)')

    args = parser.parse_args()

    process_pdf_with_json(
        args.pdf_path, args.json_path, args.output, args.dpi, args.draw_bbox,
        translate=args.translate,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_type=args.model,
        app_id=args.app_id
    )


if __name__ == "__main__":
    main()
