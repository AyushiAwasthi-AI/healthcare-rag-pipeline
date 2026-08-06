from pathlib import Path
import pytesseract
from PIL import Image
import pypdf 
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    '''
    Loads any document type and returns clean text.
    Handles: PDFs (text-based), PDFs(scanned), Images'''

    SUPPORTED_EXTENSIONS = {'.pdf','.png','.jpg', '.jpeg', '.tiff'}
    def load(self, file_path: str) -> list[dict]:
        """
        Returns LIST of page-level dicts — one dict per page.
        Each page carries its own page_number.
        Images return a single-item list for consistent interface.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() == '.pdf':
            return self._load_pdf(path)
        else:
            return [self._load_image(path)]  # wrap in list for consistent interface


    def _load_pdf(self, path: Path) -> list[dict]:
        """
        Processes PDF page by page.
        Returns one dict per page — each with its page_number.
        Skips scanned pages (< 50 chars) with a warning.
        """
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)

            if reader.is_encrypted:
                logger.warning(f"Encrypted PDF skipped: {path.name}")
                return [{
                    "text": "",
                    "source": str(path),
                    "type": "encrypted_pdf",
                    "page_number": 0,
                    "total_pages": 0,
                    "error": "PDF is encrypted"
                }]

            pages = []
            total_pages = len(reader.pages)
            scanned_count = 0

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                if len(page_text.strip()) < 50:
                    logger.info(f"Page {i+1} appears scanned in {path.name} — skipping")
                    scanned_count += 1
                    continue

                pages.append({
                    "text": page_text,
                    "source": str(path),
                    "type": "text_pdf",
                    "page_number": i + 1,      # 1-indexed, matches PDF page numbers
                    "total_pages": total_pages,
                })

            logger.info(
                f"Loaded {path.name}: "
                f"{len(pages)} pages with text, "
                f"{scanned_count} scanned pages skipped"
            )
            return pages

    def extract_pdf_text(self, path: Path) -> str:
        """Extract text from a text-based PDF."""
        text =""
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    
    def _ocr_pdf(self, path: Path) -> str:
        """OCR a scanned PDF.
        Converts each page to image first, then runs Tesseract."""

        # We'll expland this pdf2image in next iteration
        # For now, flag it clealy

        raise NotImplementedError(
            f"Scanned PDF detected: {path.name}."
            f"Install pdf2image for OCR support."
        )
    
    def _load_image(self, path: Path) -> str:
        """Run OCR directly on image files."""

        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        return {
            "text": text,
            "source": str(path),
            "type": "image_ocr"
        }
