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
    def load(self, file_path: str) -> dict:
        '''
        Main entry point. Detects file type and routes to correct extraction method.
        Returns dict with:
        -text: extracted content
        -source: original file path
        -type: how it was processed'''

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found:{file_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Route to correct extractor based on file type

        if path.suffix.lower() =='.pdf':
            return self._load_pdf(path)
        else:
            return self._load_image(path)
        
    def _load_pdf(self, path:Path) -> dict:
        """
        PDFs are either text-based or scanned images.
        We try text extraction first.
        If extracted text is too short - it's a scanned PDF.
        Route to OCR in that case."""

        '''Production-grade PDF loader.
        Handles: encrypted PDFs, scanned pages, mixed PDFs'''

        #text=self._extract_pdf_text(path)

        # If less than 50 chars extracted, PDF is likely scanned
        '''for page in reader.pages:
        page_text = page.extract_text() or ""
        if len(page_text.strip()) < 50:
        # This specific page needs OCR
        page_text = self._ocr_page(page)
        text += page_text'''

        '''if len(text.strip()) < 50:
            text = self._ocr_pdf(path)
            doc_type = "scanned_pdf"
        else:
            doc_type = "text_pdf"

        return {
            "text": text,
            "source":str(path),
            "type": doc_type
        }'''

        with open(path, 'rb') as f:
            reader=pypdf.PdfReader(f)
            #Check encryption first -  silent failure #1
            if reader.is_encrypted:
                logger.warning(f"Encrypted PDF skipped:{path.name}")
                return {
                    "text":"",
                    "source": str(path),
                    "type": "encrypted_pdf",
                    "error": "PDF is encrypted"
                }
            
            text=""
            scanned_pages = 0

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                if len(page_text.strip()) < 50:
                    #Page-level OCR fallback
                    logger.info(f"Page{i+1} appears scanned in {path.name}")
                    scanned_pages+=1
                    # OCR expansion comes next iteration
                else:
                    text+=page_text

            logger.info(
                f"Loaded {path.name}:"
                f"{len(reader.pages)} pages,"
                f"{scanned_pages} scanned,"
                f"{len(text)} chars extracted"
            )   

            return {
                "text": text,
                "source": str(path),
                "type": "mixed_pdf" if scanned_pages>0 else "text_pdf",
                "pages": len(reader.pages),
                "scanned_pages": scanned_pages
            }

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
