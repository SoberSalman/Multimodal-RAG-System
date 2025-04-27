import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Data structure for document chunks"""
    content: str
    type: str  # 'text', 'image', 'table'
    metadata: Dict[str, Any]
    embedding: Any = None  # Will be set later

class PDFExtractor:
    """Extract text, images, and tables from PDF documents"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract all content from a PDF file"""
        chunks = []
        
        # Extract using PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text
                text_chunks = self._extract_text_from_page(page, pdf_path, page_num)
                chunks.extend(text_chunks)
                
                # Extract images
                image_chunks = self._extract_images_from_page(page, pdf_path, page_num)
                chunks.extend(image_chunks)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {e}")
        
        # Extract tables using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    table_chunks = self._extract_tables_from_page(page, pdf_path, page_num)
                    chunks.extend(table_chunks)
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
        
        return chunks
    
    def _extract_text_from_page(self, page, pdf_path: str, page_num: int) -> List[DocumentChunk]:
        """Extract text from a page"""
        chunks = []
        text = page.get_text()
        
        if text.strip():
            # Simple text chunking
            paragraphs = self._split_into_paragraphs(text)
            
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    chunks.append(DocumentChunk(
                        content=paragraph,
                        type='text',
                        metadata={
                            'source': pdf_path,
                            'page': page_num + 1,
                            'chunk_id': f"text_{page_num}_{i}",
                            'type': 'text'
                        }
                    ))
        
        return chunks
    
    def _extract_images_from_page(self, page, pdf_path: str, page_num: int) -> List[DocumentChunk]:
        """Extract images from a page"""
        chunks = []
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image temporarily
                image_path = os.path.join(self.temp_dir, f"img_{page_num}_{img_index}.png")
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Extract text from image using OCR
                pil_image = Image.open(io.BytesIO(image_bytes))
                image_text = pytesseract.image_to_string(pil_image)
                
                content = image_text if image_text.strip() else f"[Image on page {page_num + 1}]"
                
                chunks.append(DocumentChunk(
                    content=content,
                    type='image',
                    metadata={
                        'source': pdf_path,
                        'page': page_num + 1,
                        'image_path': image_path,
                        'chunk_id': f"image_{page_num}_{img_index}",
                        'type': 'image',
                        'has_text': bool(image_text.strip())
                    }
                ))
                
            except Exception as e:
                logger.error(f"Error extracting image: {e}")
        
        return chunks
    
    def _extract_tables_from_page(self, page, pdf_path: str, page_num: int) -> List[DocumentChunk]:
        """Extract tables from a page"""
        chunks = []
        tables = page.extract_tables()
        
        for table_index, table in enumerate(tables):
            try:
                # Convert table to formatted text
                table_text = self._format_table(table)
                
                if table_text.strip():
                    chunks.append(DocumentChunk(
                        content=table_text,
                        type='table',
                        metadata={
                            'source': pdf_path,
                            'page': page_num + 1,
                            'chunk_id': f"table_{page_num}_{table_index}",
                            'type': 'table',
                            'rows': len(table),
                            'columns': len(table[0]) if table else 0
                        }
                    ))
            except Exception as e:
                logger.error(f"Error extracting table: {e}")
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split by double newlines or common paragraph patterns
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        result = []
        for para in paragraphs:
            if len(para) > 500:  # If paragraph is too long
                # Split by sentence
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 500:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            result.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    result.append(current_chunk.strip())
            else:
                result.append(para)
        
        return result
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format table data as structured text"""
        if not table:
            return ""
        
        # Create a markdown-style table
        result = []
        
        if len(table) > 0:
            # Headers
            headers = table[0]
            result.append(" | ".join(str(cell) for cell in headers))
            result.append(" | ".join(["---"] * len(headers)))
            
            # Data rows
            for row in table[1:]:
                result.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(result)
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.temp_dir):
                if file.startswith("img_"):
                    os.remove(os.path.join(self.temp_dir, file))
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
