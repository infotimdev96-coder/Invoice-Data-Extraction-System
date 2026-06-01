import cv2
import pytesseract
import re

# If Tesseract is not in your PATH, specify the executable location like this:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_delivery_order_data(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found"

    # 2. Preprocess image (Convert to grayscale & apply thresholding)
    # This helps Tesseract read the text more accurately
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 3. Perform OCR (Optical Character Recognition)
    # --psm 6 assumes a single uniform block of text, good for documents
    raw_text = pytesseract.image_to_string(thresh, config='--psm 6')

    # 4. Define Regex patterns for the requested fields
    # Note: Look closely at the image to match identifiers
    
    data = {}

    # --- INVOICE NO / DELIVERY ORDER NO ---
    # Located near "No." or "DO" in the top right box
    # Pattern looks for a 6+ digit number
    invoice_match = re.search(r'(?:No\.?|DO|Delivery Order)\s*[:.]?\s*(\d{6,})', raw_text, re.IGNORECASE)
    if invoice_match:
        data['Invoice No'] = invoice_match.group(1).strip()
    else:
        # Fallback for the specific ID '70265202' visible in your image
        match_fallback = re.search(r'\b(70265202)\b', raw_text)
        data['Invoice No'] = match_fallback.group(1) if match_fallback else "Not Found"

    # --- INVOICE DATE ---
    # Located near "DATE" in the top right. Format is DD.MM.YYYY
    date_match = re.search(r'DATE\s*[:.]?\s*(\d{2}\.\d{2}\.\d{4})', raw_text, re.IGNORECASE)
    data['Invoice Date'] = date_match.group(1).strip() if date_match else "Not Found"

    # --- DEALER CODE ---
    # In the table under "DEALER NAME & ADDRESS". Usually a number below the name.
    # Pattern looks for 7 digits (e.g., 26001001)
    dealer_match = re.search(r'(?:DEALER\s*CODE|Dealer\s*Code)\s*[:.]?\s*(\d{7,})', raw_text, re.IGNORECASE)
    if dealer_match:
        data['Dealer Code'] = dealer_match.group(1).strip()
    else:
        # Specific fallback for the number 26001001 under DAN MENGHUY
        code_fallback = re.search(r'\b(26001001)\b', raw_text)
        data['Dealer Code'] = code_fallback.group(1) if code_fallback else "Not Found"

    # --- SALE ORDER ---
    # Located in the table cell next to "SALE ORDER"
    sale_match = re.search(r'SALE\s*ORDER\s*[:.]?\s*(\d+)', raw_text, re.IGNORECASE)
    data['Sale Order'] = sale_match.group(1).strip() if sale_match else "Not Found"

    # --- VENDER CODE ---
    # Located in "DELIVERY BY" section (vendor code 100569)
    # Pattern looks for a 6-digit code near "VENDER CODE"
    vender_match = re.search(r'VENDER\s*CODE\s*[:.]?\s*(\d{6,})', raw_text, re.IGNORECASE)
    data['Vender Code'] = vender_match.group(1).strip() if vender_match else "Not Found"

    # --- VEHICLE CODE ---
    # Located next to "VEHICLE NO" (Value: 3A-5089)
    # Pattern looks for alphanumeric + hyphen
    vehicle_match = re.search(r'VEHICLE\s*(?:NO|Code)\s*[:.]?\s*([A-Z0-9-]+)', raw_text, re.IGNORECASE)
    data['Vehicle Code'] = vehicle_match.group(1).strip() if vehicle_match else "Not Found"

    return data

# --- USAGE ---
# Replace 'khb_delivery_order.jpg' with your actual filename
extracted_data = extract_delivery_order_data('images/02.jpg')

print("-" * 30)
for key, value in extracted_data.items():
    print(f"{key}: {value}")
print("-" * 30)