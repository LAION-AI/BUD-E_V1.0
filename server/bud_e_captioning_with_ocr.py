"""
Image Analysis Module for BUD-E

This module provides functionality for analyzing images using AI vision models.
It includes two main functions:
1. send_image_for_captioning_and_ocr: Sends an image to an AI service for captioning and OCR.
2. analyze_clipboard: Processes clipboard content, handling both text and images.

The module supports two different AI services:
- Hyper Lab API (default, uncommented)
- Groq API (commented out, can be activated by uncommenting)

To switch between services, comment out the entire block of one service and
uncomment the other. Ensure you have the correct API key set for the service you're using.
"""

import json
import base64


# ------------------------ Groq Implementation ------------------------


from groq import Groq

# Groq API Configuration
GROQ_API_KEY = "xxx"  # Replace with your Groq API key
client = Groq(api_key=GROQ_API_KEY)

def send_image_for_captioning_and_ocr(img_byte_arr, instruction):
    """
    Send an image to the Groq API for captioning and OCR.

    :param img_byte_arr: Byte array of the image
    :param instruction: Instruction for the AI model
    :return: AI-generated caption and OCR result
    """
    # Encode the image to base64
    encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')



    user_message = (
        "You are a very precise, very factual image captioning & OCR assistant. "
        "Respond using Markdown. "
        f"Instruction: {instruction}"
    )

    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

# ------------------------ Hyper Lab Implementation ------------------------
'''
import requests

# Hyper Lab API Configuration
url = 'https://api.hyprlab.io/v1/chat/completions'
API_KEY = "hypr-lab-xxxx"  # Replace with your Hyper Lab API key

def send_image_for_captioning_and_ocr(img_byte_arr, instruction):
    """
    Send an image to the Hyper Lab API for captioning and OCR.

    :param img_byte_arr: Byte array of the image
    :param instruction: Instruction for the AI model
    :return: AI-generated caption and OCR result
    """
    # Encode the image to base64
    encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a very precise, very factual image captioning & OCR assistant who follows the user's instructions.\nRespond using Markdown"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    }

    # Send request to Hyper Lab API
    response = requests.post(url, headers=headers, json=data)

    # Process and return the response
    print(response.status_code)
    print(response.text)
    response_dict = json.loads(response.text)
    return response_dict["choices"][0]["message"]["content"]

'''

# ------------------------ Common Clipboard Analysis Function ------------------------

def analyze_clipboard(client_session, instruction="Perform OCR of the image (incl. equations, tables, diagrams & maps) and then write a precise, factual description of what can be seen in the image with high details. The description after the OCR should be 10 to 40 words long."):
    """
    Process the clipboard content, handling both text and images.

    :param client_session: Dictionary containing client session data
    :param instruction: Instruction for image analysis (default provided)
    :return: Processed clipboard content as a string
    """
    print("CLIPBOARD", client_session['clipboard'][:1000])

    # Check if clipboard data exists
    if 'clipboard' not in client_session:
        return ""

    # Parse clipboard data
    clipboard_data = json.loads(client_session['clipboard'])

    # Handle text content
    if clipboard_data.get('type') == 'text':
        return "(clipboard-content)" + clipboard_data["data"] + "(/clipboard-content)"

    # Handle image content
    if isinstance(clipboard_data, dict) and clipboard_data.get('type') == 'image':
        try:
            print("Analyzing image...")
            # Decode the base64 image data
            image_data = base64.b64decode(clipboard_data['data'])

            # Send the image for captioning and OCR
            caption = send_image_for_captioning_and_ocr(image_data, instruction)

            # Print the caption in the server terminal
            print(f"Image Caption: {caption}")

            return f"(clipboard-content){caption}(/clipboard-content)"

        except Exception as e:
            print(f"Error processing clipboard image: {str(e)}")
            return ""

    # Return empty string if clipboard content is neither text nor image
    return ""