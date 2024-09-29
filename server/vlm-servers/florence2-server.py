#pip install flash_attn
# Import necessary libraries
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from transformers import AutoProcessor, AutoModelForCausalLM

# Initialize Flask app
app = Flask(__name__)

# Load Florence-2 model and processor
print("Loading Florence-2 model and processor...")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
print("Model and processor loaded successfully.")

@app.route('/caption', methods=['POST'])
def caption_image():
    print("Received caption request")
    
    # Check if the request contains the image data
    if 'image' not in request.json:
        print("Error: No image data in request")
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # Decode base64 image data
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))
        print("Image decoded successfully")

        # Prepare input for the model
        prompt = "<MORE_DETAILED_CAPTION>" 
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        print("Inputs prepared for the model")

        # Generate caption
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        print("Caption generated")

        # Process the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        print("Caption processed")

        # Return the caption
        return jsonify({"caption": parsed_answer}), 200

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5002)
