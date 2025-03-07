import os
import argparse
import glob
from openai import OpenAI
import base64
from PIL import Image
import time

# Your OpenAI API key
OPENAI_API_KEY = "api_key"  # Replace with your actual API key

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_latest_image(directory):
    """Get the most recently created image in the directory"""
    list_of_files = glob.glob(os.path.join(directory, 'marked_image_*.png'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def analyze_image(image_path, objects_to_find):
    """Analyze image using GPT-4V to find numbers associated with objects"""
    
    # Initialize OpenAI client with the API key from the script
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Create prompt
    objects_str = ", ".join(objects_to_find)
    prompt = f"""In this marked image, please identify the numbers associated with the following objects: {objects_str}.
    Please respond in this format:
    Object: [number(s)]
    Only include the objects mentioned and their associated numbers. If an object is not found, indicate 'not found'."""

    # Call GPT-4V
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Analyze marked images for specific objects')
    parser.add_argument('--objects', nargs='+', required=True,
                       help='List of objects to find in the image')
    parser.add_argument('--image', type=str,
                       help='Specific image to analyze. If not provided, uses latest from Pre_Results')
    
    args = parser.parse_args()

    # Get image path
    if args.image:
        image_path = args.image
    else:
        pre_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pre_Results")
        image_path = get_latest_image(pre_results_dir)
    
    if not image_path or not os.path.exists(image_path):
        print("No image found to analyze!")
        return

    # Analyze image
    print(f"Analyzing image: {image_path}")
    print(f"Looking for objects: {', '.join(args.objects)}")
    print("\nResults:")
    print("-" * 50)
    
    result = analyze_image(image_path, args.objects)
    print(result)

if __name__ == "__main__":
    main() 