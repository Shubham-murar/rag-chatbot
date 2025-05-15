# import os
# from typing import List, Dict, Any, Optional
# from pathlib import Path
# import fitz  # PyMuPDF for PDF processing
# import pytesseract  # OCR
# from PIL import Image
# import io
# import docx
# from bs4 import BeautifulSoup
# import requests
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import logging
# from store_in_qdrant import KnowledgeBaseManager

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Set Tesseract path for Windows
# if os.name == 'nt':  # Windows
#     try:
#         pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ACER\OneDrive\Desktop\Tesseract-OCR\tesseract.exe'
#     except Exception as e:
#         logger.warning(f"Could not set Tesseract path: {e}")

# class DocumentProcessor:
#     def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         self.kb_manager = KnowledgeBaseManager()

#     def process_file(self, file_path: str) -> List[Document]:
#         """Process a file and return a list of documents."""
#         try:
#             file_ext = os.path.splitext(file_path)[1].lower()
            
#             if file_ext == '.pdf':
#                 documents = self._process_pdf(file_path)
#             elif file_ext == '.docx':
#                 documents = self._process_docx(file_path)
#             elif file_ext == '.txt':
#                 # For text files, read the content first
#                 try:
#                     # Try different encodings
#                     encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
#                     content = None
                    
#                     for encoding in encodings:
#                         try:
#                             with open(file_path, 'rb') as file:
#                                 content = file.read().decode(encoding)
#                                 break
#                         except UnicodeDecodeError:
#                             continue
                    
#                     if content is None:
#                         raise ValueError(f"Could not decode file with any of the attempted encodings: {encodings}")
                        
#                     documents = self._process_text(content, file_path)
#                 except Exception as e:
#                     logger.error(f"Error reading text file {file_path}: {str(e)}")
#                     raise
#             else:
#                 raise ValueError(f"Unsupported file type: {file_ext}")
                
#             # Store the processed documents in Qdrant
#             try:
#                 self.kb_manager.store_text_file(file_path, self.chunk_size, self.chunk_overlap)
#                 logger.info(f"Successfully stored {file_path} in vector database")
#             except Exception as e:
#                 logger.error(f"Error storing file in vector database: {str(e)}")
                
#             return documents
                
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {str(e)}")
#             raise

#     def process_directory(self, directory_path: str) -> List[Document]:
#         """Process all supported files in a directory."""
#         directory_path = Path(directory_path)
#         if not directory_path.exists():
#             raise FileNotFoundError(f"Directory not found: {directory_path}")

#         all_documents = []
#         supported_extensions = {'.pdf', '.docx', '.txt'}

#         for file_path in directory_path.rglob('*'):
#             if file_path.suffix.lower() in supported_extensions:
#                 try:
#                     documents = self.process_file(str(file_path))
#                     all_documents.extend(documents)
#                 except Exception as e:
#                     logger.error(f"Error processing {file_path}: {str(e)}")

#         return all_documents

#     def process_url(self, url: str) -> List[Document]:
#         """Process content from a URL."""
#         try:
#             response = requests.get(url)
#             response.raise_for_status()
            
#             soup = BeautifulSoup(response.text, 'html.parser')
#             text = soup.get_text()
            
#             # Create a temporary file for the URL content
#             temp_file = Path("temp_url_content.txt")
#             temp_file.write_text(text, encoding='utf-8')
            
#             try:
#                 # Process and store using KnowledgeBaseManager
#                 self.kb_manager.store_text_file(
#                     str(temp_file),
#                     chunk_size=self.chunk_size,
#                     chunk_overlap=self.chunk_overlap
#                 )
                
#                 # Create documents for return
#                 document = Document(
#                     page_content=text,
#                     metadata={"source": url, "type": "url"}
#                 )
#                 return self.text_splitter.split_documents([document])
#             finally:
#                 # Clean up temporary file
#                 temp_file.unlink(missing_ok=True)
                
#         except Exception as e:
#             logger.error(f"Error processing URL {url}: {str(e)}")
#             raise

#     def _process_pdf(self, file_path: Path) -> List[Document]:
#         """Process PDF files with advanced features."""
#         documents = []
#         try:
#             doc = fitz.open(file_path)
            
#             for page_num in range(len(doc)):
#                 try:
#                     page = doc[page_num]
                    
#                     # Extract regular text
#                     text_content = page.get_text("text")
                    
#                     # Extract tables
#                     blocks = page.get_text("blocks")
#                     table_text = ""
#                     if blocks:
#                         for row in blocks:
#                             table_text += " | ".join(str(item) if item else "" for item in row) + "\n"
                    
#                     # Extract text from images using OCR
#                     img_list = page.get_images(full=True)
#                     image_text = ""
#                     for img_index, img in enumerate(img_list):
#                         try:
#                             xref = img[0]  # Image reference
#                             base_image = doc.extract_image(xref)
#                             image_bytes = base_image["image"]
                            
#                             # Convert image bytes to a PIL image
#                             img = Image.open(io.BytesIO(image_bytes))
                            
#                             # Extract text using Tesseract OCR
#                             image_text += pytesseract.image_to_string(img, lang="eng") + "\n"
#                         except Exception as e:
#                             logger.warning(f"Error processing image on page {page_num + 1}: {str(e)}")
#                             continue
                    
#                     # Combine all extracted text
#                     combined_text = f"{text_content}\n{table_text}\n{image_text}"
                    
#                     # Create document with metadata
#                     document = Document(
#                         page_content=combined_text,
#                         metadata={
#                             "source": str(file_path),
#                             "page": page_num + 1,
#                             "type": "pdf",
#                             "has_tables": bool(table_text),
#                             "has_images": bool(image_text)
#                         }
#                     )
#                     documents.append(document)
#                 except Exception as e:
#                     logger.error(f"Error processing page {page_num + 1}: {str(e)}")
#                     continue
            
#             doc.close()
            
#             if not documents:
#                 raise ValueError(f"No valid content extracted from {file_path}")
                
#             return self.text_splitter.split_documents(documents)
            
#         except Exception as e:
#             logger.error(f"Error processing PDF {file_path}: {str(e)}")
#             raise

#     def _process_docx(self, file_path: Path) -> List[Document]:
#         """Process DOCX files."""
#         doc = docx.Document(file_path)
#         text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
#         document = Document(
#             page_content=text,
#             metadata={"source": str(file_path), "type": "docx"}
#         )
#         return self.text_splitter.split_documents([document])

#     def _process_text(self, content: str, source_path: str) -> List[Document]:
#         """Process text content into documents."""
#         if not content or not content.strip():
#             raise ValueError("Empty or invalid text content")
            
#         chunks = self.text_splitter.split_text(content)
#         return [
#             Document(
#                 page_content=chunk,
#                 metadata={
#                     "source": source_path,
#                     "chunk_index": i,
#                     "total_chunks": len(chunks)
#                 }
#             )
#             for i, chunk in enumerate(chunks)
#             if chunk and chunk.strip()  # Only include non-empty chunks
#         ]

# def main():
#     # Example usage
#     processor = DocumentProcessor()
    
#     # Process a single file
#     # try:+
#     #     documents = processor.process_file("C:\Users\ACER\OneDrive\Desktop\AI_Clone\docs\BERT-Pre-training of Deep Bidirectional Transformers for language understanding.pdf")
#     #     print(f"Processed {len(documents)} chunks from file")
#     # except Exception as e:
#     #     print(f"Error processing file: {str(e)}")
    
#     # Process a directory
#     try:
#         documents = processor.process_directory(r"C:\Users\ACER\OneDrive\Desktop\AI_Clone\docs")
#         print(f"Processed {len(documents)} chunks from directory")
#     except Exception as e:
#         print(f"Error processing directory: {str(e)}")
    
#     # Process a URL
#     # try:
#     #     documents = processor.process_url("https://example.com")
#     #     print(f"Processed {len(documents)} chunks from URL")
#     # except Exception as e:
#     #     print(f"Error processing URL: {str(e)}")

# if __name__ == "__main__":
#     main() 