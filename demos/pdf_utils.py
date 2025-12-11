"""
PDF Utilities - Shared font loading functions

This module provides common utilities for PDF rendering operations,
particularly Unicode font loading with fallback mechanisms.
"""

import fitz
from pathlib import Path
from typing import Optional, Tuple


def load_unicode_font() -> Tuple[Optional[fitz.Font], Optional[str]]:
    """
    加载支持 Unicode 的系统字体

    Returns:
        Tuple[Optional[fitz.Font], Optional[str]]: (Font object, font file path)
        Returns (None, None) if no suitable font is found
    """
    font_candidates = [
        "C:/Windows/Fonts/seguisym.ttf",        # Segoe UI Symbol (best for symbols)
        "C:/Windows/Fonts/arialuni.ttf",        # Arial Unicode MS
        "C:/Windows/Fonts/msgothic.ttc",        # MS Gothic (good Unicode)
        "C:/Windows/Fonts/arial.ttf",           # Arial (basic)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                loaded_font = fitz.Font(fontfile=font_path)
                print(f"[Font] 使用: {font_path}")
                return loaded_font, font_path
            except Exception as e:
                print(f"[Font] 加载失败 {font_path}: {e}")
                continue

    # Fallback to built-in CJK font
    try:
        loaded_font = fitz.Font("cjk")
        print("[Font] 使用内置CJK字体")
        return loaded_font, None
    except:
        print("[Font] 警告: 无可用Unicode字体")
        return None, None


def get_safe_fontname(font: fitz.Font) -> str:
    """
    获取安全的字体名称（移除空格）

    PyMuPDF requires font names without spaces in some contexts.

    Args:
        font: fitz.Font object

    Returns:
        str: Font name with spaces removed
    """
    return font.name.replace(" ", "")
