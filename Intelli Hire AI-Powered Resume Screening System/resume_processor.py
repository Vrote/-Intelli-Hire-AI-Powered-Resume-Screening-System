import fitz
from docx import Document
import io

class ResumeProcessor:
    def extract_text_from_pdf(self, file) -> str:
        try:
            pdf_bytes = file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
            return text
        except Exception as e:
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        try:
            doc = Document(io.BytesIO(file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return ""
    
    def extract_text(self, file) -> str:
        filename = file.filename.lower()
        file.seek(0)
        
        if filename.endswith('.pdf'):
            return self.extract_text_from_pdf(file)
        elif filename.endswith('.docx'):
            return self.extract_text_from_docx(file)
        return ""