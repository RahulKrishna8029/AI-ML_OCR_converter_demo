import os
import openai
import json
import time
import requests
import fitz  # PyMuPDF
import io
from PIL import Image
import numpy as np
import cv2

# Set up OpenAI API (replace with your OpenAI API key)
openai.api_key = ''

# Azure API keys and endpoint (replace with your values)
subscription_key = ''
endpoint = ''
read_api_url = endpoint + "vision/v3.2/read/analyze"

# Function to convert PDF pages to images (one per page)
def pdf_to_images(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)  # Load each page
        pix = page.get_pixmap()  # Convert to image (pixel map)
        image_bytes = pix.tobytes()  # Convert to bytes (for OCR)
        images.append(image_bytes)

    return images

# Image preprocessing function (includes resizing, binarization, noise removal)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    # Step 1: Resize the image (scaling up)
    resized_image = image.resize((image.width * 2, image.height * 2))  # Scale up by 2x

    # Step 2: Convert to grayscale and binarize (thresholding)
    grayscale_image = resized_image.convert('L')  # Convert to grayscale
    threshold = 128
    binarized_image = grayscale_image.point(lambda p: p > threshold and 255)

    # Step 3: Remove noise using Gaussian blur
    binarized_image_cv = np.array(binarized_image)
    denoised_image = cv2.GaussianBlur(binarized_image_cv, (5, 5), 0)

    # Convert back to bytes
    denoised_image_pil = Image.fromarray(denoised_image)
    byte_array = io.BytesIO()
    denoised_image_pil.save(byte_array, format='PNG')

    return byte_array.getvalue()

# Function to send the image to Azure Read API and get the OCR results
def analyze_document(image_bytes):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream',
    }

    response = requests.post(read_api_url, headers=headers, data=image_bytes)

    if response.status_code == 202:
        operation_url = response.headers['Operation-Location']
        return operation_url
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to get the result from Azure's Read API (async nature)
def get_read_results(operation_url):
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    status = "running"

    while status == "running" or status == "notStarted":
        response = requests.get(operation_url, headers=headers)
        result = response.json()
        status = result['status']
        time.sleep(1)  # Wait for the OCR processing

    if status == "succeeded":
        return result
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to parse the OCR JSON output and extract distributed elements (words or lines)
def extract_text_elements(ocr_result):
    text_elements = []

    # Parse the pages
    for page in ocr_result.get('analyzeResult', {}).get('readResults', []):
        # Extract lines of text from each page
        for line in page.get('lines', []):
            line_text = line.get('text')
            text_elements.append(line_text)  # Add the entire line's text

    return text_elements  # Return the extracted lines as a list
# Function to enhance and structure text using OpenAI's GPT model
def enhance_and_structure_text_with_llm(page_text_elements):
    all_text_str = "\n\n".join(["\n".join(page) for page in page_text_elements])

    # Create a combined prompt for enhancement and structuring
    prompt = (
        f"Enhance and clean the following text:\n\n{all_text_str}\n\n"
        "Then, structure the text into a JSON format where each page's content is separated into individual JSON objects. "
        "Each page should have the following format: \n"
        "{\n"
        "  'page_number': <page_number>,\n"
        "  'content': <cleaned_text>\n"
        "}\n\n"
        "Provide the JSON as output."
    )

    try:
        # Make a single API call to enhance and structure text
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for enhancing and structuring text."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract and return the raw response
        raw_response = response.choices[0].message.content.strip()
        print("Raw Response:", raw_response)  # Debugging: print raw response
        return raw_response  # Return raw response as a string

    except Exception as e:
        print(f"Failed to enhance and structure text: {str(e)}")
        return None


# Main function
def main(pdf_path, output_json_path, raw_response_path):
    images = pdf_to_images(pdf_path)
    all_text_elements = []
    for i, image in enumerate(images):
        try:
            print(f"Processing page {i + 1}...")
            preprocessed_image = preprocess_image(image)
            operation_url = analyze_document(preprocessed_image)
            ocr_result = get_read_results(operation_url)
            page_text_elements = extract_text_elements(ocr_result)
            all_text_elements.append(page_text_elements)
        except Exception as e:
            print(f"Failed on page {i + 1}: {str(e)}")

    # Enhance and structure the text using a single LLM call
    raw_response = enhance_and_structure_text_with_llm(all_text_elements)

    if raw_response:
        # Save the raw response to a file
        with open(raw_response_path, 'w') as raw_response_file:
            raw_response_file.write(raw_response)
        print(f"Raw response saved to {raw_response_path}")

        # Optional: Parse raw response and save structured data (if needed)
        try:
            structured_data = json.loads(raw_response)  # Try to parse the raw response as JSON
            with open(output_json_path, 'w') as json_file:
                json.dump(structured_data, json_file, indent=4)
            print(f"Structured data saved to {output_json_path}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse raw response as JSON: {str(e)}")
    else:
        print("Failed to process and structure the text.")

# Example usage
pdf_path = ''  # Specify your PDF path here
output_json_path = ''  # Specify your output JSON path here
raw_response_path = ''  # Specify your raw response path here

main(pdf_path, output_json_path, raw_response_path)
