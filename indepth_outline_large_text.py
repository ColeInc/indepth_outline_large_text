import os
from openai import OpenAI
import textwrap
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import tiktoken

load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the tokenizer for GPT models
gpt_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(gpt_tokenizer.encode(text))


def chunk_text(text, max_tokens):
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in text.split('. '):
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_chunks(input_text, token_limit, prompt):
    output_dir = "OUT"
    current_date_time = datetime.now().strftime("%Y.%m.%d %I:%M%p").lower()
    output_file_path = os.path.join(output_dir, f"{current_date_time}.md")

    # Calculate available tokens for the chunk
    prompt_tokens = count_tokens(prompt)
    available_tokens = token_limit - prompt_tokens - \
        100  # Reserve 100 tokens for safety

    chunks = chunk_text(input_text, available_tokens)

    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            print(f"Processing chunk {
                  i+1}/{len(chunks)} - Tokens: {chunk_tokens}")

            response_text = trigger_ai_prompt_gemini(prompt, chunk, i)
            output_file.write(f"Response:\n{response_text}\n\n")


def trigger_ai_prompt_gemini(prompt, chunk, i):
    message = f"{prompt}\n\nText:\n{chunk}"
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        print(f"Error processing chunk {i+1}: {str(e)}")
        return ""


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
    # TOKEN_LIMIT = 30720  # Gemini Pro free plan token limit (30k tokens)
    TOKEN_LIMIT = 20000  # Gemini Pro free plan token limit (30k tokens)
    PROMPT = """
    Please format the following text in highly detailed bullet points.
    Bullet point the key points in batches of every few sentences.
    Please keep granular detail and don't skip any of the topics talked about.
    """

    large_input_text = fetch_input_text()
    process_chunks(large_input_text, TOKEN_LIMIT, PROMPT)
    print("Processing complete!")
