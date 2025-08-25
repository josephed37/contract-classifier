"""
Streamlit Web Application for Contract Classification with OCR.

This script creates a user-friendly web interface for the contract
classification API, now with added OCR capabilities for scanned documents.

Features:
- Handles both text-based and image-based (scanned) PDFs.
- First attempts direct text extraction with PyMuPDF for speed.
- If direct extraction yields little text, it falls back to Tesseract OCR.
- Sends the extracted text to the backend FastAPI for classification.
- Displays the predicted category and confidence score.

To run this application:
1. Ensure the Tesseract OCR engine is installed on your system.
2. Make sure the FastAPI server is running (`uvicorn src.main:app`).
3. In a new terminal, run: `streamlit run demo.py`
"""

import streamlit as st
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/classify"
MIN_TEXT_LENGTH_FOR_DIRECT_EXTRACTION = 100 # Threshold to decide if a PDF is scanned
logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---

def perform_ocr(file_bytes: bytes) -> str:
    """
    Performs OCR on a PDF file's bytes.
    
    Args:
        file_bytes: The bytes of the PDF file.

    Returns:
        A string containing the OCR-extracted text.
    """
    ocr_text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Convert page to an image (pixmap)
            pix = page.get_pixmap()
            # Convert pixmap to a PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Use pytesseract to extract text from the image
            ocr_text += pytesseract.image_to_string(img) + "\n"
    return ocr_text

def extract_text_from_pdf(uploaded_file) -> str | None:
    """
    Extracts text from an uploaded PDF, using OCR as a fallback.
    """
    try:
        file_bytes = uploaded_file.getvalue()
        
        # 1. Try direct text extraction first (fast method)
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            direct_text = "".join(page.get_text() for page in doc)

        # 2. If direct text is too short, assume it's a scanned PDF and use OCR
        if len(direct_text.strip()) < MIN_TEXT_LENGTH_FOR_DIRECT_EXTRACTION:
            st.warning("Direct text extraction yielded little content. Attempting OCR... This may take a moment.")
            with st.spinner("Performing OCR on scanned document..."):
                return perform_ocr(file_bytes)
        else:
            return direct_text
            
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error processing PDF file: {e}")
        return None

def classify_text(text: str) -> dict | None:
    """
    Sends text to the FastAPI backend for classification.
    """
    payload = {"text": text}
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        st.error(f"Failed to connect to the classification API. Ensure the backend is running at {API_URL}.")
        return None

# --- Streamlit User Interface ---

st.set_page_config(page_title="Contract Classifier", page_icon="📄")
st.title("📄 AI-Powered Contract Classifier (with OCR)")
st.markdown(
    """
    This tool classifies legal documents, including **scanned PDFs**.
    1. Upload a contract document in PDF format.
    2. Click the "Classify Contract" button.
    3. View the predicted category and the model's confidence score.
    """
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    
    if extracted_text and extracted_text.strip():
        st.success("Text successfully extracted from PDF!")
        
        with st.expander("View Extracted Text"):
            st.text_area("", extracted_text, height=250)

        if st.button("Classify Contract", type="primary"):
            with st.spinner("Sending text to the model for classification..."):
                result = classify_text(extracted_text)
            
            if result:
                st.success("Classification Complete!")
                category = result.get("predicted_category", "N/A")
                confidence = result.get("confidence_score", 0.0)
                
                st.metric(label="Predicted Contract Type", value=category)
                st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                st.info(f"The model is {confidence:.2%} confident that this document is a(n) **{category}**.")
    elif extracted_text is not None:
        st.error("Could not extract any text from the PDF, even with OCR.")
