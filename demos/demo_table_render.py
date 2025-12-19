"""
Demo: Render HTML Table to PDF using WeasyPrint

This demo demonstrates how to render an HTML table on a PDF
with automatic column width calculation and proper scaling.

Usage:
    python demos/demo_table_render.py

Output:
    output/demo_table_render.pdf
"""

import fitz  # PyMuPDF
from pathlib import Path
from io import BytesIO
from weasyprint import HTML
import time
import math


def _render_table_to_pdf(table_html, width, height, fontsize, line_height, padding):
    """
    内部函数：渲染HTML表格为PDF并返回文档和内容边界
    """
    # ============================================================
    # 表格样式配置（在这里修改颜色）
    # ============================================================
    bg_color = "#000000"      # 背景颜色：黑色
    text_color = "#ffffff"    # 文本颜色：白色
    border_color = "#ffffff"  # 边框颜色：白色
    # ============================================================

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


def render_table_with_weasyprint(page, bbox, table_html, fontsize=None):
    """
    使用WeasyPrint渲染HTML表格到PDF页面的指定区域

    高效自适应算法：只渲染2次，根据首次渲染结果直接计算最优字号。

    Args:
        page: fitz.Page对象
        bbox: Tuple of (x0, y0, x1, y1) 目标区域坐标
        table_html: HTML表格字符串
        fontsize: 基础字体大小 (pt)，如果为None则自动计算
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    print(f"\n{'='*80}")
    print(f"WeasyPrint Table Rendering:")
    print(f"  Target bbox: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
    print(f"  Size: {width:.1f} x {height:.1f} pt")
    print(f"{'='*80}\n")

    start_time = time.time()

    # 初始参数
    init_fontsize = fontsize if fontsize else 10.0
    line_height = 1.2
    padding = 2.0

    # 第1次渲染：获取初始大小
    print("  Pass 1: Measuring table size...")
    temp_doc, content_rect = _render_table_to_pdf(
        table_html, width, height, init_fontsize, line_height, padding
    )

    scale_x = width / content_rect.width if content_rect.width > 0 else 1
    scale_y = height / content_rect.height if content_rect.height > 0 else 1
    scale = min(scale_x, scale_y, 1.0)

    print(f"    fontsize={init_fontsize:.1f}pt -> content={content_rect.width:.0f}x{content_rect.height:.0f}, scale={scale:.3f}")

    # 如果表格刚好能放入bbox（scale在0.90-1.0之间），直接使用
    if 0.90 <= scale <= 1.0:
        final_doc = temp_doc
        final_rect = content_rect
        final_fontsize = init_fontsize
    else:
        temp_doc.close()

        # 计算最优字号，让表格刚好填满bbox
        # 字号与表格高度关系近似平方根（不是线性）
        # 使用 sqrt(scale) 来调整字号，更接近实际
        optimal_fontsize = init_fontsize * math.sqrt(scale) * 0.95  # 留5%余量
        optimal_line_height = max(1.0, 1.0 + (line_height - 1.0) * math.sqrt(scale))
        optimal_padding = max(0.5, padding * math.sqrt(scale))

        # 第2次渲染：使用优化后的参数
        print(f"  Pass 2: Optimized rendering...")
        final_doc, final_rect = _render_table_to_pdf(
            table_html, width, height, optimal_fontsize, optimal_line_height, optimal_padding
        )
        final_fontsize = optimal_fontsize

        scale_x = width / final_rect.width if final_rect.width > 0 else 1
        scale_y = height / final_rect.height if final_rect.height > 0 else 1
        scale = min(scale_x, scale_y, 1.0)

        print(f"    fontsize={optimal_fontsize:.1f}pt -> content={final_rect.width:.0f}x{final_rect.height:.0f}, scale={scale:.3f}")

    # 计算最终渲染尺寸
    actual_width = final_rect.width * scale
    actual_height = final_rect.height * scale
    target_rect = fitz.Rect(x0, y0, x0 + actual_width, y0 + actual_height)

    print(f"\n  Final: fontsize={final_fontsize:.1f}pt, scale={scale:.3f}")
    print(f"  Rendered size: {actual_width:.1f} x {actual_height:.1f} pt")

    # 将渲染结果嵌入到目标页面
    page.show_pdf_page(target_rect, final_doc, 0, clip=final_rect)

    final_doc.close()

    elapsed_time = time.time() - start_time
    print(f"[OK] WeasyPrint table rendering complete (耗时: {elapsed_time:.3f}s)\n")


def main():
    """Main function to demonstrate table rendering"""
    print("\n" + "="*80)
    print("HTML Table Rendering Demo")
    print("="*80)

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 用户提供的表格数据
    table_data = {
        "bbox": [346.0, 320.0, 1355.0, 1195.0],
        "text": """<table><thead><tr><th>Type</th><th>Model</th><th>FID↓</th><th>IS↑</th><th>Pre↑</th><th>Rec↑</th><th>#Para</th><th>#Step</th><th>Time</th></tr></thead><tbody><tr><td>GAN</td><td>BigGAN [13]</td><td>6.95</td><td>224.5</td><td><strong>0.89</strong></td><td>0.38</td><td>112M</td><td>1</td><td>—</td></tr><tr><td>GAN</td><td>GigaGAN [42]</td><td>3.45</td><td>225.5</td><td>0.84</td><td><strong>0.61</strong></td><td>569M</td><td>1</td><td>—</td></tr><tr><td>GAN</td><td>StyleGan-XL [74]</td><td>2.30</td><td>265.1</td><td>0.78</td><td>0.53</td><td>166M</td><td>1</td><td>0.3 [74]</td></tr><tr><td>Diff.</td><td>ADM [26]</td><td>10.94</td><td>101.0</td><td>0.69</td><td>0.63</td><td>554M</td><td>250</td><td>168 [74]</td></tr><tr><td>Diff.</td><td>CDM [36]</td><td>4.88</td><td>158.7</td><td>—</td><td>—</td><td>—</td><td>8100</td><td>—</td></tr><tr><td>Diff.</td><td>LDM-4-G [70]</td><td>3.60</td><td>247.7</td><td>—</td><td>—</td><td>400M</td><td>250</td><td>—</td></tr><tr><td>Diff.</td><td>DiT-L/2 [63]</td><td>5.02</td><td>167.2</td><td>0.75</td><td>0.57</td><td>458M</td><td>250</td><td>31</td></tr><tr><td>Diff.</td><td>DiT-XL/2 [63]</td><td>2.27</td><td>278.2</td><td>0.83</td><td>0.57</td><td>675M</td><td>250</td><td>45</td></tr><tr><td>Diff.</td><td>L-DiT-3B [3]</td><td>2.10</td><td>304.4</td><td>0.82</td><td>0.60</td><td>3.0B</td><td>250</td><td>>45</td></tr><tr><td>Diff.</td><td>L-DiT-7B [3]</td><td>2.28</td><td>316.2</td><td>0.83</td><td>0.58</td><td>7.0B</td><td>250</td><td>>45</td></tr><tr><td>Mask.</td><td>MaskGIT [17]</td><td>6.18</td><td>182.1</td><td>0.80</td><td>0.51</td><td>227M</td><td>8</td><td>0.5 [17]</td></tr><tr><td>Mask.</td><td>RCG (cond.) [51]</td><td>3.49</td><td>215.5</td><td>—</td><td>—</td><td>502M</td><td>20</td><td>1.9 [51]</td></tr><tr><td>AR</td><td>VQVAE-2<sup>†</sup> [68]</td><td>31.11</td><td>~45</td><td>0.36</td><td>0.57</td><td>13.5B</td><td>5120</td><td>—</td></tr><tr><td>AR</td><td>VQGAN<sup>†</sup> [30]</td><td>18.65</td><td>80.4</td><td>0.78</td><td>0.26</td><td>227M</td><td>256</td><td>19 [17]</td></tr><tr><td>AR</td><td>VQGAN [30]</td><td>15.78</td><td>74.3</td><td>—</td><td>—</td><td>1.4B</td><td>256</td><td>24</td></tr><tr><td>AR</td><td>VQGAN-re [30]</td><td>5.20</td><td>280.3</td><td>—</td><td>—</td><td>1.4B</td><td>256</td><td>24</td></tr><tr><td>AR</td><td>ViTVQ [92]</td><td>4.17</td><td>175.1</td><td>—</td><td>—</td><td>1.7B</td><td>1024</td><td>>24</td></tr><tr><td>AR</td><td>ViTVQ-re [92]</td><td>3.04</td><td>227.4</td><td>—</td><td>—</td><td>1.7B</td><td>1024</td><td>>24</td></tr><tr><td>AR</td><td>RQTran. [50]</td><td>7.55</td><td>134.0</td><td>—</td><td>—</td><td>3.8B</td><td>68</td><td>21</td></tr><tr><td>AR</td><td>RQTran.-re [50]</td><td>3.80</td><td>323.7</td><td>—</td><td>—</td><td>3.8B</td><td>68</td><td>21</td></tr><tr><td>VAR</td><td>VAR-d16</td><td>3.30</td><td>274.4</td><td>0.84</td><td>0.51</td><td>310M</td><td>10</td><td>0.4</td></tr><tr><td>VAR</td><td>VAR-d20</td><td>2.57</td><td>302.6</td><td>0.83</td><td>0.56</td><td>600M</td><td>10</td><td>0.5</td></tr><tr><td>VAR</td><td>VAR-d24</td><td>2.09</td><td>312.9</td><td>0.82</td><td>0.59</td><td>1.0B</td><td>10</td><td>0.6</td></tr><tr><td>VAR</td><td>VAR-d30</td><td>1.92</td><td>323.1</td><td>0.82</td><td>0.59</td><td>2.0B</td><td>10</td><td>1</td></tr><tr><td>VAR</td><td>VAR-d30-re<br>(validation data)</td><td><strong>1.73</strong></td><td><strong>350.2</strong></td><td>0.82</td><td>0.60</td><td>2.0B</td><td>10</td><td>1</td></tr></tbody></table>"""
    }

    # 解析bbox（从200dpi图像坐标转换为PDF坐标）
    # PDF默认72dpi，图像200dpi，转换系数 = 72/200 = 0.36
    dpi_scale = 72 / 200
    img_x0, img_y0, img_x1, img_y1 = table_data["bbox"]
    x0 = img_x0 * dpi_scale
    y0 = img_y0 * dpi_scale
    x1 = img_x1 * dpi_scale
    y1 = img_y1 * dpi_scale

    print(f"  Image coords (200dpi): [{img_x0}, {img_y0}, {img_x1}, {img_y1}]")
    print(f"  PDF coords (72dpi):    [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")

    # 打开原始PDF
    source_pdf_path = "test-source-table.pdf"
    doc = fitz.open(source_pdf_path)
    page = doc[0]

    print(f"  Source PDF: {source_pdf_path}")
    print(f"  PDF page size: {page.rect.width:.1f} x {page.rect.height:.1f} pt")

    print("\nRendering table on original PDF...")
    print("-" * 80)

    bbox = (x0, y0, x1, y1)
    render_table_with_weasyprint(page, bbox, table_data["text"])  # 自动计算最优字号

    # 第二个表格
    table_data_2 = {
        "bbox": [860.0, 1658.0, 1412.0, 1953.0],
        "text": """<table><thead><tr><th>Type</th><th>Model</th><th>FID↓</th><th>IS↑</th><th>Time</th></tr></thead><tbody><tr><td>GAN</td><td>BigGAN [13]</td><td>8.43</td><td>177.9</td><td>—</td></tr><tr><td>Diff.</td><td>ADM [26]</td><td>23.24</td><td>101.0</td><td>—</td></tr><tr><td>Diff.</td><td>DiT-XL/2 [63]</td><td>3.04</td><td>240.8</td><td>81</td></tr><tr><td>Mask.</td><td>MaskGIT [17]</td><td>7.32</td><td>156.0</td><td>0.5†</td></tr><tr><td>AR</td><td>VQGAN [30]</td><td>26.52</td><td>66.8</td><td>25†</td></tr><tr><td>VAR</td><td>VAR-d36-s</td><td><strong>2.63</strong></td><td><strong>303.2</strong></td><td>1</td></tr></tbody></table>"""
    }

    img_x0_2, img_y0_2, img_x1_2, img_y1_2 = table_data_2["bbox"]
    x0_2 = img_x0_2 * dpi_scale
    y0_2 = img_y0_2 * dpi_scale
    x1_2 = img_x1_2 * dpi_scale
    y1_2 = img_y1_2 * dpi_scale

    print(f"\n  Table 2 - Image coords (200dpi): {table_data_2['bbox']}")
    print(f"  Table 2 - PDF coords (72dpi): [{x0_2:.1f}, {y0_2:.1f}, {x1_2:.1f}, {y1_2:.1f}]")

    bbox_2 = (x0_2, y0_2, x1_2, y1_2)
    render_table_with_weasyprint(page, bbox_2, table_data_2["text"])

    # Save PDF
    output_path = output_dir / "test-table-rendered.pdf"
    doc.save(str(output_path))
    doc.close()

    print("\n" + "="*80)
    print(f"[OK] PDF saved to: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
