import openai
import json
import re
from azure_vision_read_OCR import OpenAIkey
# Set up OpenAI API (replace with your OpenAI API key)
openai.api_key = OpenAIkey

# Function to load text from a file
def load_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to process text via an LLM call and structure it as JSON
def structure_text_to_json(file_path):
    # Step 1: Load the text from the file
    raw_text = load_text_from_file(file_path)

    # Step 2: Prepare the LLM prompt
    prompt = (
        "Please extract relevant sections of the following text and structure it as a JSON object. "
        "The JSON should include any structured information such as names, dates, addresses, and other key-value pairs. "
        f"Here is the text:\n\n{raw_text}\n\n"
        "Format the response in valid JSON."
    )

    try:
        # Step 3: Call the LLM API
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for extracting and structuring information."},
                {"role": "user", "content": prompt}
            ]
        )

        # Step 4: Extract the JSON from the response
        raw_response = response.choices[0].message.content.strip()
        print("Raw LLM Response:", raw_response)  # Debugging: print the raw response

        # Use regex to extract JSON content
        json_match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```', raw_response, re.DOTALL)

        if json_match:
            # Clean the JSON string and validate it
            json_data = json.loads(json_match.group(1))
            return json_data
        else:
            print("No valid JSON format found in the response.")
            return None

    except Exception as e:
        print(f"Failed to structure text to JSON: {str(e)}")
        return None

# Function to store JSON data into a file
def save_json_to_file(json_data, output_file):
    try:
        with open(output_file, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"JSON data successfully saved to {output_file}")
    except Exception as e:
        print(f"Failed to save JSON to file: {str(e)}")

# Example usage
file_path = "/Users/grahulkrishna/Downloads/raw_response.txt"  # Replace with the path to your text file
output_file = "/Users/grahulkrishna/Downloads/structured_data.json"     # Specify the output JSON file path

json_data = structure_text_to_json(file_path)
if json_data:
    save_json_to_file(json_data, output_file)
