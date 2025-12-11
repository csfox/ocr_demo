"""
Create test PDF and JSON for demo_text_overlay.py
"""

import fitz
import json


def create_test_pdf():
    """Create a simple test PDF with text and formulas"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add title
    title_rect = fitz.Rect(50, 50, 545, 100)
    page.draw_rect(title_rect, color=(0.9, 0.95, 1.0), fill=(0.9, 0.95, 1.0))
    page.insert_textbox(
        title_rect,
        "Sample Document for Text Overlay Demo",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    # Add text block 1
    text1_rect = fitz.Rect(50, 120, 545, 180)
    page.draw_rect(text1_rect, color=(1.0, 1.0, 0.9), fill=(1.0, 1.0, 0.9))
    page.insert_textbox(
        text1_rect,
        "Einstein discovered that energy and mass are related.",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)
    )

    # Add text block 2 with formula
    text2_rect = fitz.Rect(50, 200, 545, 260)
    page.draw_rect(text2_rect, color=(0.9, 1.0, 0.9), fill=(0.9, 1.0, 0.9))
    page.insert_textbox(
        text2_rect,
        "The famous equation is energy equals mass times speed of light squared.",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)
    )

    # Add text block 3
    text3_rect = fitz.Rect(50, 280, 545, 340)
    page.draw_rect(text3_rect, color=(1.0, 0.9, 1.0), fill=(1.0, 0.9, 1.0))
    page.insert_textbox(
        text3_rect,
        "Newton's second law describes motion with force equals mass times acceleration.",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)
    )

    # Add text block 4 with numbers
    text4_rect = fitz.Rect(50, 360, 545, 420)
    page.draw_rect(text4_rect, color=(0.95, 0.95, 0.95), fill=(0.95, 0.95, 0.95))
    page.insert_textbox(
        text4_rect,
        "The year 2024 marks 100 years since the theory was published.",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)
    )

    # Save PDF
    output_path = "output/test_input.pdf"
    doc.save(output_path)
    doc.close()

    print(f"[SUCCESS] 测试PDF已创建: {output_path}")
    return output_path


def create_test_json():
    """Create corresponding JSON with OCR results"""
    ocr_data = {
        "pages": [
            {
                "page_index": 0,
                "elements": [
                    {
                        "bbox": [50, 50, 545, 100],
                        "category": "title",
                        "text": "Sample Document for Text Overlay Demo"
                    },
                    {
                        "bbox": [50, 120, 545, 180],
                        "category": "text",
                        "text": "Einstein discovered that energy and mass are related."
                    },
                    {
                        "bbox": [50, 200, 545, 260],
                        "category": "text",
                        "text": "The famous equation is $E=mc^2$ which is energy equals mass times speed of light squared."
                    },
                    {
                        "bbox": [50, 280, 545, 340],
                        "category": "text",
                        "text": "Newton's second law describes motion with $F=ma$ which is force equals mass times acceleration."
                    },
                    {
                        "bbox": [50, 360, 545, 420],
                        "category": "text",
                        "text": "The year 2024 marks 100 years since the theory was published."
                    }
                ]
            }
        ]
    }

    output_path = "output/test_input.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] 测试JSON已创建: {output_path}")
    return output_path


if __name__ == "__main__":
    print("\n创建测试文件...")
    print("=" * 60)
    pdf_path = create_test_pdf()
    json_path = create_test_json()
    print("=" * 60)
    print("\n运行demo命令:")
    print(f"  uv run python demos/demo_text_overlay.py {pdf_path} {json_path}")
    print()
