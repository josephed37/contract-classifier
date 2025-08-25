# -*- coding: utf-8 -*-
"""
Streamlit Web Application for Contract Classification with OCR and XAI.

This script uses a simple, single-column layout and a robust, button-driven
workflow. It connects to a FastAPI backend to provide classification and
explainability for uploaded PDF documents.

This version includes a reduced sample count for LIME to prevent system crashes
on machines with limited RAM.
"""

import streamlit as st
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# --- Configuration ---
API_URL_CLASSIFY = "http://127.0.0.1:8000/classify"
API_URL_EXPLAIN_BATCH = "http://127.0.0.1:8000/classify-explain-batch"
CLASS_NAMES = ['Employment', 'NDA', 'Partnership', 'SLA', 'Vendor']

logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(_uploaded_file_bytes):
    """Extracts text from an uploaded PDF, using OCR as a fallback."""
    try:
        with fitz.open(stream=_uploaded_file_bytes, filetype="pdf") as doc:
            direct_text = "".join(page.get_text() for page in doc)
        if len(direct_text.strip()) < 100:
            st.warning("Low text count; attempting OCR.")
            return perform_ocr(_uploaded_file_bytes)
        return direct_text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def perform_ocr(file_bytes: bytes) -> str:
    """Performs OCR on a PDF file's bytes."""
    ocr_text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text += pytesseract.image_to_string(img) + "\n"
    return ocr_text

def classify_text(text: str) -> dict | None:
    """Gets a single prediction from the /classify endpoint."""
    payload = {"text": text}
    try:
        response = requests.post(API_URL_CLASSIFY, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the classification API. Is the backend running?")
        return None

def get_probabilities_for_lime(texts: list[str]) -> np.ndarray:
    """Prediction function for LIME. Calls the batch-explain endpoint in manageable chunks."""
    all_probs = []
    chunk_size = 64

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        valid_texts_in_chunk = [t for t in chunk if t and not t.isspace()]
        if not valid_texts_in_chunk:
            all_probs.extend([[1.0 / len(CLASS_NAMES)] * len(CLASS_NAMES)] * len(chunk))
            continue
        payload = {"texts": valid_texts_in_chunk}
        try:
            response = requests.post(API_URL_EXPLAIN_BATCH, json=payload, timeout=180)
            response.raise_for_status()
            probs_list = response.json().get("all_probabilities", [])
            result_map = {text: probs for text, probs in zip(valid_texts_in_chunk, probs_list)}
            for original_text in chunk:
                probs_dict = result_map.get(original_text)
                if probs_dict:
                    probs = [probs_dict.get(name, 0.0) for name in CLASS_NAMES]
                    all_probs.append(probs)
                else:
                    all_probs.append([1.0 / len(CLASS_NAMES)] * len(CLASS_NAMES))
        except requests.exceptions.RequestException as e:
            st.error(f"API request for explanation failed: {e}")
            return np.array([[1.0 / len(CLASS_NAMES)] * len(CLASS_NAMES)] * len(texts))
    return np.array(all_probs)

# --- Streamlit User Interface ---

st.set_page_config(page_title="Contract Classifier", page_icon="📄")
st.title("📄 AI-Powered Contract Classifier")

if 'result' not in st.session_state: st.session_state.result = None
if 'text' not in st.session_state: st.session_state.text = None

st.header("1. Upload Your Document")
uploaded_file = st.file_uploader(
    "Upload a contract in PDF format. The tool handles both text-based and scanned documents.",
    type="pdf"
)

if uploaded_file is not None:
    if st.button("Classify Contract", type="primary", use_container_width=True):
        with st.spinner("Processing PDF..."):
            file_bytes = uploaded_file.getvalue()
            st.session_state.text = extract_text_from_pdf(file_bytes)
        if st.session_state.text:
            with st.spinner("Classifying document..."):
                st.session_state.result = classify_text(st.session_state.text)
        else:
            st.error("Could not extract any text from the PDF.")
            st.session_state.result = None

if st.session_state.result:
    st.markdown("---")
    st.header("2. Classification Result")
    result = st.session_state.result
    category = result.get("predicted_category", "N/A")
    confidence = result.get("confidence_score", 0.0)
    
    st.metric("Predicted Category", category)
    st.progress(confidence, f"Confidence: {confidence:.2%}")
    st.info(f"The model is **{confidence:.2%}** confident that this document is a(n) **{category}**.")

    if st.button("🔍 Explain Prediction", use_container_width=True):
        with st.spinner("Generating explanation... This may take up to a minute."):
            explainer = LimeTextExplainer(class_names=CLASS_NAMES)
            explanation = explainer.explain_instance(
                st.session_state.text,
                get_probabilities_for_lime,
                num_features=10,
                num_samples=1000,
                labels=[CLASS_NAMES.index(category)]
            )
            st.markdown("---")
            st.header("3. Prediction Explanation")
            # --- THE FIX ---
            # This text is now more general and accurate.
            st.markdown("The highlighted words are the most influential features for this prediction. Words supporting the predicted category are on the right, while words against it are on the left.")
            components.html(explanation.as_html(), height=350, scrolling=True)

if st.session_state.text:
    with st.expander("View Full Extracted Text"):
        st.text_area("", st.session_state.text, height=250)
