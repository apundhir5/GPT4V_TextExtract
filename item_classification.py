import spacy
import os
from openai import OpenAI
# import openai
from dotenv import load_dotenv

# Load the spaCy model
nlp = spacy.load("en_core_web_md")
load_dotenv(dotenv_path=".env")
api_key=os.getenv("OPENAI_API_KEY")

# List of eligible items defined using spaCy's Doc objects for better semantic comparison
eligible_terms = [nlp(text) for text in ["cough syrup", "ibuprofen", "bandaid"]]

def is_medically_reimbursable(item, eligible_terms):
    # Process the item with spaCy
    item_doc = nlp(item.lower())
    
    # Determine if the item matches any eligible term using semantic similarity
    # Set a threshold for similarity; here it is set to 0.5, adjust based on testing
    threshold = 0.55
    for term in eligible_terms:
        if item_doc.similarity(term) > threshold:
            return True, item_doc.similarity(term)
    return False, 0

# Function to parse the responses and extract only the 'yes' or 'no' part
def parse_response(response_list):
    answers = []
    for response in response_list:
        # Split on '-' and strip spaces, then get the last part which is the answer
        answer = response.split('-')[-1].strip()
        answers.append(answer)
    return answers


def is_medically_reimbursable_openai(items):
    prompt = "Determine if the following grocery items are eligible for medical reimbursement based on their medical necessity or health benefits:\n"
    for index, item in enumerate(items, 1):
        prompt += f"{index}. {item}\n"
    prompt += "\nProvide a simple 'yes' or 'no' answer for each item, respectively."
   

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    answers = response.choices[0].message.content.strip().split('\n')
    # print(answers)

    # Map answers back to items
    # item_answer_dict = {item: answer.strip().split(': ')[-1].lower() == 'yes' for item, answer in zip(items, answers)}
    
    item_answer_dict = {}
    for item, answer in zip(items, answers):
        # Normalize the answer by removing leading numbers and trimming whitespace
        normalized_answer = answer.strip().split('. ')[-1].strip().lower()
        # Handle different formats using '-' or ':' delimiters or direct yes/no after numeric prefix
        if ' - ' in normalized_answer:
            answer_parts = normalized_answer.split(' - ')
        elif ': ' in normalized_answer:
            answer_parts = normalized_answer.split(': ')
        else:
            answer_parts = [normalized_answer]

        # Determine if the item is eligible based on 'yes'
        is_eligible = answer_parts[-1] == 'yes'
        item_answer_dict[item] = is_eligible

    return item_answer_dict


receipt_items = [
    "Whole wheat bread",
    "Sugar-free cough syrup",
    "Organic apple",
    "Ibuprofen 200mg",
    "Regular soda",
    "Toothpaste"
]

print("Using Spacy for item similiarity\n")
for item in receipt_items:
    result, similarity = is_medically_reimbursable(item, eligible_terms)
    print(f"Item: {item}, Similarity: {similarity}, Eligible for reimbursement: {result}")

print("\n\nUsing Generative AI for item similiarity")
results = is_medically_reimbursable_openai(receipt_items)
for item, is_eligible in results.items():
    print(f"Item: {item}, Eligible for reimbursement: {is_eligible}")



