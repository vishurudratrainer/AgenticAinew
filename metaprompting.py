import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"   # or any model you installed via `ollama pull`

import requests
import json


def ollama_chat(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()["response"]




def improve_prompt(bad_prompt):
    meta_prompt = f"""
You are a prompt-engineering assistant.
Improve the user prompt by making it:
– clearer
– more detailed
– with constraints
– with better structure

User prompt:
\"\"\"{bad_prompt}\"\"\"
"""

    improved = ollama_chat(meta_prompt)
    print("Improved Prompt:\n", improved)
    return improved

# Test
improve_prompt("Write code for a chatbot")
