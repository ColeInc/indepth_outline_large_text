import os
from anthropic import Anthropic
import textwrap
from datetime import datetime
from dotenv import load_dotenv
import tiktoken
import time

load_dotenv()

# Configure Anthropic API
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize the tokenizer for Claude models
claude_tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude uses the same tokenizer as GPT models


def count_tokens(text):
    return len(claude_tokenizer.encode(text))


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

    # Calculate available tokens for the chunk, being more conservative
    prompt_tokens = count_tokens(prompt)
    # Reserve more tokens for the response and system overhead
    available_tokens = min(token_limit // 2, 4000)  # Using half of limit or 4000, whichever is smaller
    
    chunks = chunk_text(input_text, available_tokens)

    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            print(f"Processing chunk {i+1}/{len(chunks)} - Tokens: {chunk_tokens}")
            
            # Add delay between requests to avoid overloading
            if i > 0:
                time.sleep(2)  # 2 second delay between requests

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response_text = trigger_ai_prompt_claude(prompt, chunk, i)
                    if response_text:
                        break
                    time.sleep(5)  # Wait longer if we got an empty response
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(5)  # Wait between retry attempts
            
            seperator = "################\n################\n################\n################\n################\n################\n\n"
            output_file.write(f"Response:\n{seperator}{response_text}\n\n")


def trigger_ai_prompt_claude(prompt, chunk, i):
    # TODO: remove this early return, just for prompt testing
    # TODO: remove this early return, just for prompt testing
    # TODO: remove this early return, just for prompt testing
    message = f"{prompt}\n\nText:\n{chunk}"
    try:
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20240620",  # Updated to Sonnet model
            max_tokens=3000,  # More conservative limit for Sonnet
            messages=[{
                "role": "user",
                "content": message
            }]
        )
        return response.content[0].text
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
    TOKEN_LIMIT = 16000  # Claude 3 Sonnet has a 32k context window
    # PROMPT = """
    # Please format the following text in highly detailed bullet points.
    # Bullet point the key points in batches of every few sentences.
    # Please keep granular detail and don't skip any of the topics talked about.
    # """

    # PROMPT = """
    # Please give a highly detailed bullet point summary of the following text. bullet point all key points mentioned in each paragraph. for each bullet point, please make it a minimum of 2 sentences:
    # """

    # PROMPT = """
    # I'm going to paste a video transcript in with some timestamps below. could you please create a similar style of timestamps, outlining what is talked about in each of the corresponding timestamps? please, reading the timestamps specified, try to outline what is happening at least every 2 minutes. please give your answers from the tense of something like "the speaker talks about the importance of xyz".
    # """

    # PROMPT = """
    # Please summarise the following text. summarise it using an approach where you look at the text approximately every 3 paragraphs at a time, and create some bullet points summarise what was spoken about in that part of the text. please continue to do this throughout the entire text, outputting your answer as you go.
    # """

    # PROMPT = """
    # i'm going to paste some text below. please look at it from the perspective of a student trying to extract the most useful note taking and key take away points and things to learn from this text. please title each of the key points you identify, and give a breakdown of each one.
    # """

    PROMPT = """
    i'm going to paste a transcript below. based on this text i want you to identify each of the main startup ideas that are talked about. for each startup idea, generate a concise, single-page business plan for a startup. The plan should be clear, straight to the point, and avoid unnecessary verbosity. Follow this structure:
Executive Summary – Briefly summarize the business idea, including its mission and core purpose.
Business Description – Explain what the business does, the problem it solves, and the unique value it offers.
Market Analysis – Identify the target audience, key market trends, and main competitors.
Marketing Plan – Outline how the business will attract and retain customers.
Operations Plan – Summarize the key operational aspects, including location, suppliers, and processes.
Management & Team – Highlight the leadership team and their relevant expertise.
Financial Plan – Provide key financial projections, including revenue streams, expected costs, and profitability timeline.
Ensure the response is structured, professional, and easy to read. The tone should be clear, compelling, and investor-friendly. Keep the entire response under 500 words for efficiency.
transcript:
    """

    large_input_text = fetch_input_text()
    process_chunks(large_input_text, TOKEN_LIMIT, PROMPT)
    print("Processing complete!")
