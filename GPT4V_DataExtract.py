from openai import OpenAI
import base64
import json
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

load_dotenv(dotenv_path=".env")

image_local = './Images/1.jpg'
image_url = f"data:image/jpeg;base64,{encode_image(image_local)}"
# image_url = 'https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/02/19/ML-1955-2.jpg'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_system_promot():
    template: str = """
       You are an expert at information extraction from images of receipts.

       Given this of a receipt, extract the following information:
       - The name and address of the vendor
       - The names and costs of each of the items that were purchased
       - The date and time that the receipt was issued. This must be formatted like 'MM/DD/YY HH:MM'
       - The subtotal (i.e. the total cost before tax)
       - The tax rate
       - The total cost after tax

       Do not guess. If some information is missing just return "N/A" in the relevant field.
       If you determine that the image is not of a receipt, just set all the fields in the formatting instructions to "N/A". 
       
       You must obey the output format under all circumstances. Please follow the formatting instructions exactly.
       Do not return any additional comments or explanation. 
       Return the data in json format.
       """
    return template

def call_genai():
    response = client.chat.completions.create(
        model='gpt-4-vision-preview', 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_system_promot()},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ],
            }
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content

def save_json(json_data):
    # filename_without_extension = os.path.splitext(os.path.basename(urlparse(image_url).path))[0] #for URL image
    filename_without_extension = os.path.splitext(os.path.basename(image_local))[0] #for local image
    json_filename = f"{filename_without_extension}.json"

    with open("./Data/" + json_filename, 'w') as file:
        json.dump(json_data, file, indent=4)

    print(f"JSON data saved to {json_filename}")


json_string = call_genai()
json_string = json_string.replace("```json\n", "").replace("\n```", "")

json_data = json.loads(json_string)
save_json(json_data)

