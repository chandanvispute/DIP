import sys
import os
import cv2
import pytesseract
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from PIL import Image, ImageDraw, ImageFont
import textwrap
import re

nltk.download('punkt', quiet=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Image not found or cannot be read."
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"

def separate_email_parts(text):
    """
    Separate an email into header, body, and signature.
    - Header: All lines until the first blank line.
    - Signature: If the body contains lines starting with common signature markers, these are treated as the signature.
    """
    lines = text.splitlines()
    header = []
    rest = []
    blank_found = False

    for line in lines:
        if not blank_found:
            if line.strip() == "":
                blank_found = True
            else:
                header.append(line)
        else:
            rest.append(line)
    
    main_text = "\n".join(rest).strip()
    
    signature_markers = ["thank you", "thanks", "regards", "best", "sincerely"]
    main_lines = main_text.splitlines()
    signature_index = None
    for i in range(len(main_lines)-1, -1, -1):
        for marker in signature_markers:
            if main_lines[i].strip().lower().startswith(marker):
                signature_index = i
                break
        if signature_index is not None:
            break
    
    if signature_index is not None:
        body = "\n".join(main_lines[:signature_index]).strip()
        signature = "\n".join(main_lines[signature_index:]).strip()
    else:
        body = main_text
        signature = ""
    
    header_text = "\n".join(header).strip()
    return header_text, body, signature

def summarize_text_body(text, num_sentences=1):
    """Summarize the given text using Sumy's LSA summarizer."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, num_sentences)
        summarized = " ".join(str(sentence) for sentence in summary_sentences)
        return summarized
    except Exception as e:
        return f"Error summarizing text: {e}"

def summarize_email(text, num_sentences=1):
    """
    For emails, preserve header and signature, but summarize only the body.
    """
    header, body, signature = separate_email_parts(text)
    summarized_body = summarize_text_body(body, num_sentences)
    parts = []
    if header:
        parts.append(header)
    if summarized_body:
        parts.append(summarized_body)
    if signature:
        parts.append(signature)
    return "\n\n".join(parts)

def create_summary_image(summary, output_path, mode="general"):
    try:
        width = 800
        height = 600
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            header_font = ImageFont.truetype("arialbd.ttf", size=26) 
            body_font = ImageFont.truetype("arial.ttf", size=24)
            signature_font = ImageFont.truetype("ariali.ttf", size=24) 
        except Exception:
            header_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            signature_font = ImageFont.load_default()
        
        y_text = 20
        line_spacing = 10

        if mode.lower() == "email":
            header_text, body_text, signature_text = separate_email_parts(summary)
            if header_text:
                header_lines = textwrap.wrap(header_text, width=60)
                for line in header_lines:
                    draw.text((20, y_text), line, fill=(0, 0, 0), font=header_font)
                    bbox = draw.textbbox((0, 0), line, font=header_font)
                    text_height = bbox[3] - bbox[1]
                    y_text += text_height + line_spacing
                y_text += 10
                draw.line((20, y_text, width - 20, y_text), fill=(200,200,200), width=2)
                y_text += 20
            if body_text:
                body_lines = textwrap.wrap(body_text, width=60)
                for line in body_lines:
                    draw.text((20, y_text), line, fill=(0, 0, 0), font=body_font)
                    bbox = draw.textbbox((0, 0), line, font=body_font)
                    text_height = bbox[3] - bbox[1]
                    y_text += text_height + line_spacing
                y_text += 20
            if signature_text:
                draw.line((20, y_text, width - 20, y_text), fill=(200,200,200), width=2)
                y_text += 20
                signature_lines = textwrap.wrap(signature_text, width=60)
                for line in signature_lines:
                    draw.text((20, y_text), line, fill=(0, 0, 0), font=signature_font)
                    bbox = draw.textbbox((0, 0), line, font=signature_font)
                    text_height = bbox[3] - bbox[1]
                    y_text += text_height + line_spacing
        else:
            all_lines = textwrap.wrap(summary, width=60)
            for line in all_lines:
                draw.text((20, y_text), line, fill=(0, 0, 0), font=body_font)
                bbox = draw.textbbox((0, 0), line, font=body_font)
                text_height = bbox[3] - bbox[1]
                y_text += text_height + line_spacing

        img.save(output_path)
    except Exception as e:
        print(f"Error creating summary image: {e}")

if __name__ == "__main__":
    try:
        image_path = r"C:\Users\alkas\Downloads\sample.jpg"
        
        if not os.path.exists(image_path):
            print(f"Error: The file {image_path} does not exist.")
            sys.exit(1)
        
        mode = "general"  
        
        text = extract_text(image_path)
        print("Extracted Text:", text)
        
        sentences = nltk.sent_tokenize(text)
        original_count = len(sentences)
        desired_count = max(1, int(round(original_count * 0.35)))
        
        if mode.lower() == "email":
            summary = summarize_email(text, desired_count)
        else:
            summary = summarize_text_body(text, desired_count)
        
        if not summary.strip():
            summary = text
        
        print("Summarized Text:", summary)
        
        base = os.path.basename(image_path)
        summary_image_path = os.path.join(os.path.dirname(image_path), "summarized_" + base)
        create_summary_image(summary, summary_image_path, mode)
        print("Summary Image:", summary_image_path)
    except Exception as e:
        print("Error in summarize_text.py:", e)
