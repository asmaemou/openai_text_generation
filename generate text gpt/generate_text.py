import os
import openai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Please add it to the .env file.")

# Initialize OpenAI API with the key
openai.api_key = api_key

def generate_text(prompt):
    try:
        # Make a request to OpenAI's new ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Replace with "gpt-4" if you have access
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # Adjust the creativity level if needed
        )

        # Extract and return the generated text
        return response['choices'][0]['message']['content']

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    # Get user input
    user_input = input("Enter a prompt: ")
    # Generate and print the response
    result = generate_text(user_input)
    print(f"Generated text: {result}")
