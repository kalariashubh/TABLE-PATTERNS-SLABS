import os
import json

from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


def detect_pattern(pdf_path, temp_folder):
    """
    Detect slab schedule pattern (1 to 5)
    based strictly on HEADER structure.
    """

    image_paths = convert_pdf_to_images(pdf_path, temp_folder)

    if not image_paths:
        raise Exception("No image generated for pattern detection.")

    first_image = image_paths[0]

    classification_prompt = """
You are an expert at identifying RCC slab schedule header patterns.

Look ONLY at the HEADER structure.
Ignore data rows completely.

There are EXACTLY 5 slab patterns.

Return ONLY one number:
1
2
3
4
5

No explanation.
No extra text.

=========================
PATTERN 1
=========================

Columns:
- TYPE
- THICKNESS
- STEEL ALONG SPAN
- STEEL ACROSS SPAN
- REMARKS

=========================
PATTERN 2
=========================

Columns:
- TYPE
- THICKNESS
- STEEL ALONG SHORT SPAN
- STEEL ACROSS SHORT SPAN
- REMARKS

=========================
PATTERN 3
=========================

Columns:
- SLAB MARKED
- THK.
- TYPE
- REINFORCEMENT
    - SHORT BAR
    - LONG BAR
- REMARKS

=========================
PATTERN 4
=========================

Columns:
- SLAB NO.
- SLAB THK.
- MIX
- TYPE
- MAIN STEEL
    - 11 TO SHORT SPAN
    - 11 TO LONG SPAN
- EX. TOP OVER SUPPORT
    - 11 TO SHORT SPAN
    - 11 TO LONG SPAN
- DIST. STEEL

=========================
PATTERN 5
=========================

Columns:
- BAR MARKED
- DIA
- SPACING
- REMARKS

=========================

Return ONLY the number.
"""

    result = extract_from_image(first_image, classification_prompt)
    result = result.strip()

    if not result.isdigit():
        raise Exception(f"Pattern detection failed. Model returned: {result}")

    return int(result)
