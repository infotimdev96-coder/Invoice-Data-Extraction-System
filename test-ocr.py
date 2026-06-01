import cv2
import pytesseract

# If Tesseract is not in your system PATH, specify the executable path here:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_invoice(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply adaptive thresholding to deal with the background color
    # This helps separate the dark text from the pink/red background
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Define languages: 'eng' for English, 'khm' for Khmer
    # Combining them allows Tesseract to read both scripts in the document
    custom_config = r'--oem 3 --psm 5 -l eng+khm'
    
    # 5. Run OCR
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return extracted_text

# Example Usage:
image_file = '/Users/timdev/Downloads/image.jpeg'
try:
    text = extract_text_from_invoice(image_file)
    print("--- Extracted Text ---")
    print(text)
except Exception as e:
    print(f"An error occurred: {e}")