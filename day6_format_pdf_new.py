# ********** IMPORT LIBRARY **************
from typing import Dict, List
import os
from dotenv import load_dotenv
from load_pdf import load_pdf
import logging
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any


class DocumentPreprocess:
    def __init__(self, pdf_url, document_name: str = None, document_id: str = None):
        self.pdf_url = pdf_url
        self.html_content = ""
        self.soup = None
        self.document_name = document_name
        self.document_id = document_id
        
    def extract_html_from_pdf(self) -> str:
        """Extract the full HTML content from the PDF using PyMuPDF (fitz)"""
        pdf_document = load_pdf(self.pdf_url)
        if not pdf_document:
            print("PDF document is not loaded correctly. Exiting extraction process.")
            return ""
            
        full_html = ""
        try:
            for page_num, page in enumerate(pdf_document):
                page_text = page.get_textpage().extractXHTML()
                full_html += page_text + "\n"
            print("Successfully extracted HTML from PDF.")
            self.html_content = full_html
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
        except Exception as e:
            print(f"Error while extracting HTML from PDF. Error: {e}")
            
        return full_html

    def _create_section(self, content: List[str], headers: Dict[str, str]) -> Dict[str, Any]:
        """Create a section with content and metadata"""
        return {
            'page_content': '\n'.join(content) if content else "",
            'metadata': {
                'header1': headers['header1'],
                'header2': headers['header2'],
                'header3': headers['header3'],
                'header4': headers['header4'],
                'document_name': self.document_name,
                'document_id': self.document_id
            }
        }

    def _get_header_level(self, element) -> int:
        """Determine the header level of an element"""
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            return int(element.name[1])
            
        elif element.name == 'p':
            # bold_content = element.find('b')
            # if bold_content:
          text = element.get_text().strip()
          if re.match(r'^(\d+\.)*\d+\s+\w+', text): # Matches patterns like "1.2.3 Title"
            header_level = text.count('.') + 1
            return min(header_level, 4)  # Cap at level 4
        
        elif element.name == 'p':
          text = element.get_text().strip()
          if len(text) < 35:
              return 1 
        
        return 0
      
    def process_document(self) -> List[Dict[str, Any]]:
        """Process the document and return structured output with headers and content"""
        if not self.html_content:
            print("No HTML content available. Run extract_html_from_pdf first.")
            return []

        if not self.soup:
            self.soup = BeautifulSoup(self.html_content, 'html.parser')

        elements = self.soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        processed_content = []
        current_headers = {
            'header1': None,
            'header2': None,
            'header3': None,
            'header4': None
        }
        current_content = []

        for element in elements:
            header_level = self._get_header_level(element)
            
            if header_level > 0:
                # Save current content if exists
                if current_content:
                    processed_content.append(self._create_section(current_content, current_headers))
                    current_content = []

                # Update headers
                header_text = element.get_text().strip()
                current_headers[f'header{header_level}'] = header_text
                
                # Clear lower-level headers
                for i in range(header_level + 1, 5):
                    current_headers[f'header{i}'] = ""
            else:
                # Add non-header content
                current_content.append(element.get_text().strip())

        # Add final section if there's remaining content
        if current_content:
            processed_content.append(self._create_section(current_content, current_headers))

        return processed_content


if __name__ == "__main__":
  url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"

  processor = DocumentPreprocess(
    pdf_url=url,
    document_name="Attention Is All You Need",
    document_id="111"
  )

  processor.extract_html_from_pdf()
  structured_content = processor.process_document()
  print(structured_content[:10])
  
 

