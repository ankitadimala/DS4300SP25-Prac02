import fitz
import re
import os 

def get_pdf_text(file_path):
    """
    Extracts text from all pages of the PDF
    """
    pdf = fitz.open(file_path)
    text = ""
    for page in pdf: 
        # using "text" method to get text of the page
        page = page.get_text("text")
        text += page + "\n"
    
    return text 

def clean_text(text): 
    """
    Cleans extracted pdf text: removes whitespace, slide numbers, etc.
    """
    # Collapse multiple newlines and trim overall text.
    text = re.sub(r'\n+', '\n', text).strip()
    # Remove lines with only digits (standalone slide numbers)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove headers (like "DS 4300")
    text = re.sub(r'DS\s*4300', '', text, flags=re.IGNORECASE)
    # Remove isolated bullet lines (lines that are just bullet markers)
    text = re.sub(r'^\s*[-–•*]+\s*$', '', text, flags=re.MULTILINE)
    # Remove extraneous reference lines (e.g., lines starting with "Figure from:" or "Reference:")
    text = re.sub(r'^(Figure from:|Reference:).*\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove repeated headers (e.g., multiple "ACID Properties")
    text = re.sub(r'(ACID Properties\s*){2,}', 'ACID Properties\n', text, flags=re.IGNORECASE)
    # Remove lines that contain only question marks
    text = re.sub(r'^[\?]+\s*$', '', text, flags=re.MULTILINE)

    return text

def process_pdf(file_path):
    """
    Processes a single PDF file (extracting and cleaning the text)
    """
    raw_text = get_pdf_text(file_path)
    cleaned_text = clean_text(raw_text)
    
    return cleaned_text

def process_folder(folder_path):
    """
    Processes all PDF files in a given folder.
    Returns: 
        data (Dict): dictionary mapping cleaned files to PDF filenames
    """
    data = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            cleaned_text = process_pdf(file_path)
            data[filename] = cleaned_text

    return data

if __name__ == "__main__":

    folder_path = "/Users/nubahaahsan/Desktop/DS4300/Practicals/DS4300SP25-Prac02/data"
    cleaned_data = process_folder(folder_path)
    for filename, text in cleaned_data.items(): 
        print(f"Processed {filename} with {len(text)} characters")

