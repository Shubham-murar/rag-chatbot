import fitz  # PyMuPDF for PDF processing
import pytesseract  # OCR
from PIL import Image
import io
import os

# 🔹 Set Tesseract OCR path (Windows users must install it)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ACER\OneDrive\Desktop\Tesseract-OCR\tesseract.exe'

# 🔹 Path to PDFs folder
PDF_FOLDER = "docs"
OUTPUT_FILE = "docs/knowledge_base.txt"


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file (including tables)."""
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text("text") + "\n"  # Extract text content

        # Extract tables if available
        blocks = page.get_text("blocks")
        if blocks:
            for row in blocks:
                text += " | ".join(str(item) if item else "" for item in row) + "\n"

    return text


def extract_text_from_images(pdf_path):
    """Extracts text from images inside a PDF using OCR."""
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]  # Image reference
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert image bytes to a PIL image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Extract text using Tesseract OCR
            text += pytesseract.image_to_string(img, lang="eng") + "\n"

    return text


def extract_all_pdfs():
    """Processes all PDFs and saves extracted text to knowledge_base.txt."""
    all_text = ""

    for pdf_file in os.listdir(PDF_FOLDER):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            print(f"📄 Processing: {pdf_file}")

            text_content = extract_text_from_pdf(pdf_path)
            image_text = extract_text_from_images(pdf_path)

            # Append extracted data to the final knowledge base
            all_text += f"\n\n### Extracted from {pdf_file} ###\n\n"
            all_text += text_content
            all_text += image_text

    # Save extracted text to a file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(all_text)

    print("✅ Extraction complete! Knowledge base saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    extract_all_pdfs()
