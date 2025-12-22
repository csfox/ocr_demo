"""
JSON Text Translator Module

Translates text elements in OCR JSON files while preserving LaTeX formulas.
- Inline formulas ($...$) are replaced with placeholders {F_n} during translation
- Display formulas ($$...$$) and formula category elements are skipped entirely

Usage:
    python json_translator.py input.json output.json --src en --tgt zh
"""

import json
import re
import argparse
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# Translation API configuration
TRANSLATION_API_CONFIG = {
    "base_url": "http://172.16.199.171:24861",
    "endpoint": "/api/v1/text/translate",
    "timeout": 120,  # 2 minutes timeout
    "default_model": "deepseek_v3",
    "default_app_id": "ocr_demo",
}


def extract_inline_formulas(text: str) -> Tuple[str, List[str]]:
    """
    Extract inline formulas ($...$) and replace with placeholders {F_n}.

    Args:
        text: Input text with inline formulas

    Returns:
        Tuple of (text_with_placeholders, list_of_formulas)

    Example:
        Input: "The equation $E=mc^2$ shows energy."
        Output: ("The equation {F_0} shows energy.", ["$E=mc^2$"])
    """
    formulas = []

    def replace_formula(match):
        formula = match.group(0)
        index = len(formulas)
        formulas.append(formula)
        return f"{{F_{index}}}"

    # Match inline formulas $...$ (not $$...$$)
    # Use negative lookbehind/lookahead to avoid matching $$
    pattern = r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'
    text_with_placeholders = re.sub(pattern, replace_formula, text)

    return text_with_placeholders, formulas


def restore_inline_formulas(text: str, formulas: List[str]) -> str:
    """
    Restore inline formulas from placeholders.

    Args:
        text: Text with placeholders {F_n}
        formulas: List of original formulas

    Returns:
        Text with formulas restored
    """
    for i, formula in enumerate(formulas):
        # Handle possible variations in placeholder format after translation
        # Some translators might add spaces or change formatting
        patterns = [
            f"{{F_{i}}}",           # Exact match
            f"{{ F_{i} }}",         # With spaces
            f"{{F_ {i}}}",          # Space before number
            f"{{ F_ {i} }}",        # Multiple spaces
            f"\\{{F_{i}\\}}",       # Escaped braces
        ]
        for pattern in patterns:
            text = text.replace(pattern, formula)

    return text


def is_display_formula(text: str) -> bool:
    """
    Check if text is a display formula ($$...$$).

    Args:
        text: Text to check

    Returns:
        True if text is wrapped in $$...$$
    """
    text = text.strip()
    return text.startswith("$$") and text.endswith("$$")


def should_skip_translation(text: str, category: str) -> bool:
    """
    Determine if an element should skip translation.

    Args:
        text: Element text
        category: Element category

    Returns:
        True if element should be skipped
    """
    # Skip formula category
    if category.lower() == "formula":
        return True

    # Skip display formulas
    if is_display_formula(text):
        return True

    # Skip empty text
    if not text or not text.strip():
        return True

    # Skip pure numbers
    if re.match(r"^\d+$", text.strip()):
        return True

    # Skip pure placeholders
    if re.match(r"^\{F_\d+\}$", text.strip()):
        return True

    return False


def translate_text(
    text: str,
    src_lang: str = "en",
    tgt_lang: str = "zh",
    model_type: str = None,
    app_id: str = None,
    domain: str = None,
) -> str:
    """
    Translate text using the translation API.

    Args:
        text: Text to translate
        src_lang: Source language code
        tgt_lang: Target language code
        model_type: Model type (deepseek_v3, volcengine, etc.)
        app_id: Application ID
        domain: Domain hint

    Returns:
        Translated text, or original text if translation fails
    """
    if not text or not text.strip():
        return text

    # Use defaults if not specified
    model_type = model_type or TRANSLATION_API_CONFIG["default_model"]
    app_id = app_id or TRANSLATION_API_CONFIG["default_app_id"]

    # Build request payload
    payload = {
        "model_type": model_type,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "text": text,
        "app_id": app_id,
    }

    if domain:
        payload["domain"] = domain

    # Build URL
    url = TRANSLATION_API_CONFIG["base_url"] + TRANSLATION_API_CONFIG["endpoint"]

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=TRANSLATION_API_CONFIG["timeout"],
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()

        if result.get("code") == "0000":
            translated = result.get("data", {}).get("result", "")
            if translated:
                return translated
            else:
                print(f"  [WARNING] Empty translation result, using original text")
                return text
        else:
            error_code = result.get("code", "unknown")
            error_desc = result.get("desc", "Unknown error")
            print(f"  [ERROR] Translation API error: {error_code} - {error_desc}")
            return text

    except requests.exceptions.Timeout:
        print(f"  [ERROR] Translation request timeout")
        return text
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Translation request failed: {e}")
        return text
    except Exception as e:
        print(f"  [ERROR] Unexpected error during translation: {e}")
        return text


def translate_element_text(
    text: str,
    category: str,
    src_lang: str = "en",
    tgt_lang: str = "zh",
    model_type: str = None,
    app_id: str = None,
    domain: str = None,
) -> str:
    """
    Translate a single element's text, preserving inline formulas.

    Args:
        text: Element text
        category: Element category
        src_lang: Source language
        tgt_lang: Target language
        model_type: Translation model
        app_id: Application ID
        domain: Domain hint

    Returns:
        Translated text with formulas preserved
    """
    # Check if should skip
    if should_skip_translation(text, category):
        return text

    # Extract inline formulas and replace with placeholders
    text_with_placeholders, formulas = extract_inline_formulas(text)

    # If text is now empty or just placeholders, skip translation
    text_without_placeholders = re.sub(r'\{F_\d+\}', '', text_with_placeholders).strip()
    if not text_without_placeholders:
        return text

    # Translate
    translated_text = translate_text(
        text_with_placeholders,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model_type=model_type,
        app_id=app_id,
        domain=domain,
    )

    # Restore formulas
    if formulas:
        translated_text = restore_inline_formulas(translated_text, formulas)

    return translated_text


def translate_json(
    input_path: str,
    output_path: str,
    src_lang: str = "en",
    tgt_lang: str = "zh",
    model_type: str = None,
    app_id: str = None,
    domain: str = None,
) -> Dict:
    """
    Translate all text elements in a JSON file.

    Args:
        input_path: Input JSON file path
        output_path: Output JSON file path
        src_lang: Source language
        tgt_lang: Target language
        model_type: Translation model
        app_id: Application ID
        domain: Domain hint

    Returns:
        Statistics dictionary
    """
    print("\n" + "=" * 80)
    print("JSON Text Translation")
    print("=" * 80)
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Languages: {src_lang} -> {tgt_lang}")
    print(f"  Model: {model_type or TRANSLATION_API_CONFIG['default_model']}")
    print("=" * 80 + "\n")

    # Load JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support two JSON formats
    if 'pages' in data:
        pages_data = data['pages']
    elif 'page_index' in data and 'elements' in data:
        pages_data = [data]
    else:
        print("[ERROR] Unrecognized JSON format!")
        return {"error": "Invalid JSON format"}

    # Statistics
    stats = {
        "total_elements": 0,
        "translated_elements": 0,
        "skipped_elements": 0,
        "formulas_preserved": 0,
    }

    # Process each page
    for page_data in pages_data:
        page_index = page_data.get('page_index', 0)
        elements = page_data.get('elements', [])

        print(f"[Page {page_index}] Processing {len(elements)} elements...")

        for elem_idx, element in enumerate(elements):
            text = element.get('text', '')
            category = element.get('category', 'text')

            stats["total_elements"] += 1

            if should_skip_translation(text, category):
                stats["skipped_elements"] += 1
                continue

            # Count formulas before translation
            _, formulas = extract_inline_formulas(text)
            stats["formulas_preserved"] += len(formulas)

            # Translate
            print(f"  [{elem_idx + 1}/{len(elements)}] Translating {category}...", end=" ")
            translated_text = translate_element_text(
                text, category,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                model_type=model_type,
                app_id=app_id,
                domain=domain,
            )

            if translated_text != text:
                element['text'] = translated_text
                stats["translated_elements"] += 1
                print("OK")
            else:
                print("(unchanged)")

    # Save output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "=" * 80)
    print("Translation Statistics:")
    print(f"  Total elements: {stats['total_elements']}")
    print(f"  Translated: {stats['translated_elements']}")
    print(f"  Skipped: {stats['skipped_elements']}")
    print(f"  Formulas preserved: {stats['formulas_preserved']}")
    print(f"  Output saved to: {output_path}")
    print("=" * 80 + "\n")

    return stats


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Translate OCR JSON text while preserving LaTeX formulas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python json_translator.py input.json output.json
  python json_translator.py input.json output.json --src en --tgt zh
  python json_translator.py input.json output.json --model volcengine

Supported models:
  - deepseek_v3 (default)
  - volcengine
        """
    )

    parser.add_argument('input', type=str, help='Input JSON file path')
    parser.add_argument('output', type=str, help='Output JSON file path')
    parser.add_argument('--src', type=str, default='en', help='Source language (default: en)')
    parser.add_argument('--tgt', type=str, default='zh', help='Target language (default: zh)')
    parser.add_argument('--model', type=str, default=None, help='Translation model (deepseek_v3, volcengine)')
    parser.add_argument('--app-id', type=str, default=None, help='Application ID')
    parser.add_argument('--domain', type=str, default=None, help='Domain hint')

    args = parser.parse_args()

    translate_json(
        args.input,
        args.output,
        src_lang=args.src,
        tgt_lang=args.tgt,
        model_type=args.model,
        app_id=args.app_id,
        domain=args.domain,
    )


if __name__ == "__main__":
    main()
