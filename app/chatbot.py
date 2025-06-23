import logging
import os
from dotenv import load_dotenv
from app.loader import load_docx_chunks
from app.embedder import build_index, search
from groq import Groq
import re
import html

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Setup Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load and index chunks
chunks = None
index = None


def generate_response(query, context):
    prompt = f"""
You are SLMS360 Assistant — a helpful chatbot built to guide users through the SLMS360 Admin Portal.

Your job is to:
- Give clear, concise answers to user questions
- Use natural, friendly language
- Avoid repititive greetings
- Use relevant emojis where they help make things clearer or more fun
- Break down steps simply when needed using • bullets
- Avoid technical jargon unless absolutely necessary
- Do NOT use symbols like #, *, ** or --- — use • instead for any lists

Below is the context from the system followed by the user’s question.

Use it to generate a short, engaging, and helpful response in a casual tone.

Context:
{context}

User Query:
{query}
"""

    logging.info("Sending request to Groq...")

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024
        )
        reply = response.choices[0].message.content.strip()
        logging.info(f"Groq Response: {reply}")
        formatted = sanitize_and_format(reply)
        return formatted

    except Exception as e:
        logging.error(f"Groq SDK Error: {str(e)}")
        return "Sorry, an error occurred while generating the response."


def handle_query(query):
    logging.info(f"User query: {query}")

    matched_chunks = search(query, index, chunks)
    context = "\n\n".join(matched_chunks)

    logging.info("Matched chunks:")
    for chunk in matched_chunks:
        logging.info(f"---\n{chunk}\n")

    return generate_response(query, context)

# def initialize_embeddings(doc_path="./SLMATE.docx"):
def initialize_embeddings(doc_path="SLMATE.docx"):
    global chunks, index
    logging.info("Loading and indexing document...")
    chunks = load_docx_chunks(doc_path)
    index, _ = build_index(chunks)


def sanitize_and_format(text: str) -> str:
    """Sanitize Groq output for HTML and convert markdown-style formatting."""

    # Convert markdown formatting BEFORE escaping
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)   # Bold
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)               # Italic (asterisks)
    text = re.sub(r"_(.*?)_", r"<em>\1</em>", text)                 # Italic (underscores)
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)             # Inline code

    # Then escape the rest (except tags)
    text = html.escape(text)

    # Unescape allowed tags
    allowed_tags = ["<strong>", "</strong>", "<em>", "</em>", "<code>", "</code>"]
    for tag in allowed_tags:
        escaped = html.escape(tag)
        text = text.replace(escaped, tag)

    # Convert newlines to <br> for HTML display
    text = text.replace("\n", "<br>")

    return text

