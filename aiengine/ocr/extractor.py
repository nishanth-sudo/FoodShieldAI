import re

from PIL import Image

try:
    import numpy as np
except ImportError:
    np = None


FIELD_PATTERNS = {
    "product_name": [
        r"(?:product|name)[:\s]*([A-Z][A-Za-z\s\-]+)",
        r"^(?:[A-Z][A-Za-z\s\-]{3,})$",
    ],
    "brand": [
        r"(?:brand|by)[:\s]*([A-Z][A-Za-z\s\-]+)",
    ],
    "expiry_date": [
        r"(?:exp(?:iry)?|use\s*by|best\s*before|sell\s*by|bb)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(?:exp(?:iry)?|use\s*by|best\s*before|sell\s*by|bb)[:\s]*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})",
        r"(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4})",
        r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    ],
    "ingredients": [
        r"(?:ingredients?)[:\s]*(.+?)(?:\d+[gml]|nutrition|allergen|$)",
    ],
    "net_weight": [
        r"(\d+[\.,]?\d*\s*(?:g|kg|ml|l|oz|lb)s?\.?)",
        r"(?:net\s*(?:wt|weight)|weight)[:\s]*(\d+[\.,]?\d*\s*(?:g|kg|ml|l))",
    ],
    "nutritional_info": [
        r"(?:nutrition|per\s*(?:100|serving))",
    ],
    "allergens": [
        r"(?:allergen|contains)[:\s]*(.+?)(?:\.|$)",
    ],
}

# Minimum number of parsed fields from regex before trying Ollama fallback
_REGEX_MIN_FIELDS = 2

# Prompt used when Ollama fallback is invoked
_OCR_PARSE_PROMPT = """Extract structured information from this food product label text.
Return ONLY a valid JSON object with these fields (omit fields not found):
{{
  "product_name": "<product name>",
  "brand": "<brand name>",
  "expiry_date": "<expiry/best-before date as found>",
  "ingredients": "<ingredients list>",
  "net_weight": "<weight or volume>",
  "allergens": "<allergen info>",
  "nutritional_info": "<any nutrition facts>"
}}

Label text:
{raw_text}"""


class LabelTextExtractor:
    """
    OCR + label field extractor.

    Backends:
      - ``paddleocr``  — PaddleOCR (recommended, high accuracy)
      - ``easyocr``    — EasyOCR
      - ``tesseract``  — pytesseract
      - ``mock``       — Hard-coded fixture for testing

    Ollama fallback:
      When regex parsing yields fewer than ``ollama_min_fields`` fields,
      the raw OCR text is sent to a local Ollama model for intelligent
      field extraction. Set ``ollama_model=None`` to disable.

    Args:
        backend:          OCR engine backend name.
        ollama_model:     Ollama text model for parse fallback
                          (default: ``"llama3.2:3b"``). Pass ``None`` to disable.
        ollama_min_fields: Minimum regex-parsed fields needed to skip Ollama
                          fallback (default: 2).
    """

    def __init__(
        self,
        backend: str = "paddleocr",
        ollama_model: str | None = "llama3.2:3b",
        ollama_min_fields: int = _REGEX_MIN_FIELDS,
    ) -> None:
        self.backend_name = backend
        self._ollama_model = ollama_model
        self._ollama_min_fields = ollama_min_fields
        self._ocr = None
        self._ollama_client = None

        if backend != "mock":
            self._init_backend()
        if ollama_model:
            self._init_ollama(ollama_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: Image.Image) -> dict:
        raw_text = self._run_ocr(image)
        parsed = self._parse_fields(raw_text)

        # Attempt Ollama-powered smart parsing when regex result is sparse
        if self._ollama_client is not None and len(parsed) < self._ollama_min_fields:
            parsed = self._ollama_parse(raw_text, parsed)

        return {
            "raw_text": raw_text,
            "parsed": parsed,
            "backend": self.backend_name,
        }

    def extract_from_label_region(
        self, image: Image.Image, label_region: Image.Image | None = None
    ) -> dict:
        return self.extract(label_region or image)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        if self.backend_name == "paddleocr":
            try:
                from paddleocr import PaddleOCR

                self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except ImportError:
                raise ImportError(
                    "paddleocr is required. Install with: pip install paddleocr"
                ) from None
        elif self.backend_name == "easyocr":
            try:
                import easyocr

                self._ocr = easyocr.Reader(["en"])
            except ImportError:
                raise ImportError(
                    "easyocr is required. Install with: pip install easyocr"
                ) from None
        elif self.backend_name == "tesseract":
            try:
                import pytesseract

                self._ocr = pytesseract
            except ImportError:
                raise ImportError(
                    "pytesseract is required. Install with: pip install pytesseract"
                ) from None
        else:
            raise ValueError(f"Unsupported OCR backend: {self.backend_name}")

    def _init_ollama(self, model: str) -> None:
        try:
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model=model, temperature=0.1, max_tokens=500)
            if client.is_available:
                self._ollama_client = client
            else:
                import logging
                logging.getLogger(__name__).info(
                    "Ollama not available — OCR fallback parsing disabled. "
                    "Start Ollama and pull model '%s' to enable.",
                    model,
                )
        except Exception:
            pass  # Ollama unavailable — silently degrade to regex-only

    # ------------------------------------------------------------------
    # OCR engine
    # ------------------------------------------------------------------

    def _run_ocr(self, image: Image.Image) -> str:
        if self.backend_name == "mock":
            return self._mock_ocr()
        img_array = np.array(image) if np else image
        if self.backend_name == "paddleocr":
            result = self._ocr.ocr(img_array, cls=True)
            texts = []
            for line in result:
                for word_info in line:
                    texts.append(word_info[1][0])
            return " ".join(texts)
        elif self.backend_name == "easyocr":
            result = self._ocr.readtext(img_array)
            return " ".join(item[1] for item in result)
        elif self.backend_name == "tesseract":
            return self._ocr.image_to_string(image)
        return ""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_fields(self, text: str) -> dict:
        parsed = {}
        for field, patterns in FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text.strip(), re.IGNORECASE)
                if match:
                    parsed[field] = (
                        match.group(1).strip() if match.lastindex else match.group(0).strip()
                    )
                    break
        return parsed

    def _ollama_parse(self, raw_text: str, regex_parsed: dict) -> dict:
        """Use Ollama to extract label fields when regex confidence is low."""
        if not raw_text.strip():
            return regex_parsed

        prompt = _OCR_PARSE_PROMPT.format(raw_text=raw_text[:2000])  # cap context length
        result = self._ollama_client.chat_json(prompt)  # type: ignore[union-attr]
        if result:
            # Merge: keep any regex hits, fill gaps with Ollama output
            merged = {**result, **regex_parsed}
            merged["_parsed_by"] = "ollama_fallback"
            return merged

        return regex_parsed

    # ------------------------------------------------------------------
    # Mock fixture
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_ocr() -> str:
        return (
            "Product: Fresh Whole Milk\n"
            "Brand: DairyBest\n"
            "Ingredients: Pasteurized Whole Milk, Vitamin D3\n"
            "Net Wt: 1 Liter (33.8 fl oz)\n"
            "Best Before: 2026-08-15\n"
            "Nutrition Facts: Calories 150, Total Fat 8g, Protein 8g\n"
            "Allergens: Contains Milk"
        )
