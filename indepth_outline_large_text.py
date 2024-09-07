import os
from openai import OpenAI
import textwrap
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text, chunk_size):
    return textwrap.wrap(text, chunk_size, break_long_words=False, replace_whitespace=False)

def process_chunks(input_text, token_limit, prompt):
    # Estimate chunk size (assuming average token is ~4 characters)
    estimated_chunk_size = token_limit * 4
    output_dir = "OUT"

    # Get the current date time for filename
    current_date_time = datetime.now().strftime("%Y.%m.%d %I:%M%p").lower()

    # Construct the output file path with the current date appended
    output_file_path = os.path.join(
       output_dir, f"{current_date_time}.txt")

    # Chunk the input text
    chunks = chunk_text(input_text, estimated_chunk_size)
    
    # Process each chunk
    # with open('gemini_responses.txt', 'w') as output_file:
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # response_text = trigger_ai_prompt(prompt, chunk, i)
            response_text = trigger_ai_prompt_v2(prompt, chunk, i)
            output_file.write(f"Chunk {i+1} Response:\n{response_text}\n\n")

def trigger_ai_prompt(prompt, chunk, i):
    # Prepare the message for Gemini
            message = f"{prompt}\n\nText:\n{chunk}"
            print("message going out\n", message)
            
            try:
                # Send request to Gemini API
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(message)
                print("response back:\n", response.text)
                # response = model.generate_content(f"{prompt}\n\n{message}")
                
                # Write response to file
                return response.text
                # output_file.write(f"Chunk {i+1} Response:\n{response.text}\n\n")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")

def trigger_ai_prompt_v2(prompt, chunk, i):
    # Prepare the message for OpenAI
    message = f"{prompt}\n\nText:\n{chunk}"
    # print("message going out\n", message)
    
    # try:
    #     # Send request to OpenAI API
    #     response = openai.Completion.create(
    #         engine="text-davinci-002",  # or another appropriate engine
    #         prompt=message,
    #         max_tokens=1000  # adjust as needed
    #     )
    #     # print("response back:\n", response.choices[0].text)
    #     print("response back:\n", response)
        
    #     # Write response to file
    #     # output_file.write(f"Chunk {i+1} Response:\n{response.choices[0].text}\n\n")
    #     return response.choices[0].text

    # except Exception as e:
    #     print(f"Error processing chunk {i+1}: {str(e)}")


    try:
        # Make a request to the OpenAI API
        response = client.chat.completions.create(
            # model="gpt-4o",  # You can change this to other models like "gpt-3.5-turbo"
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": message},
                {"role": "user", "content": chunk}
            ],
            max_tokens=2000
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def fetch_input_text():
    try:
        with open('INPUT.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("Error: INPUT.txt file not found.")
        return ""
    except IOError:
        print("Error: Unable to read INPUT.txt file.")
        return ""


if __name__ == "__main__":    
    # Define token limit and prompt
    TOKEN_LIMIT = 2000  # Adjust based on the actual Gemini API token limit
    PROMPT = """
    Please format the following text in highly detailed bullet points.
    Bullet point the key points in batches of every few sentences.
    Please keep granular detail and don't skip any of the topics talked about.
    """

    # Fetch input text from file
    large_input_text = fetch_input_text()
    # Process the text
    process_chunks(large_input_text, TOKEN_LIMIT, PROMPT)
    # print("cole", os.getenv("GEMINI_API_KEY"))

    print("Processing complete!")

    
