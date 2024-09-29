"""
BUD-E Language Model Interface

This module provides interfaces to various Language Model (LLM) APIs and local setups.
It includes implementations for Groq, Hyperlapse, local VLLM/Ollama, and Together AI.
Users can choose which implementation to use by commenting/uncommenting the relevant sections.

The main function `ask_LLM` is consistent across all implementations, allowing for easy switching between different LLM providers.
"""

# ------------------ Groq API Implementation ------------------
from groq import Groq
import json

API_KEY = "xxxxxx"  # Replace with your Groq API key

client = Groq(api_key=API_KEY)

def ask_LLM(modelname, systemprompt, content, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.1, presence_penalty=1.1, streaming=True):
    """
    Send a request to the Groq API and return the LLM's response.

    :param modelname: Name of the model to use (e.g., "llama-3.1-70b-versatile")
    :param systemprompt: System prompt to set the context
    :param content: User's input content
    :param temperature: Controls randomness in generation
    :param top_p: Controls diversity of generation
    :param max_tokens: Maximum number of tokens to generate
    :param frequency_penalty: Penalizes frequent tokens
    :param presence_penalty: Penalizes repeated tokens
    :param streaming: Whether to use streaming response
    :return: LLM's response as a string
    """
    print(modelname)
    
    # Construct the messages for the API request
    messages = [
        {"role": "system", "content": systemprompt},
        {"role": "user", "content": content}
    ]
    
    try:
        if streaming:
            # Handle streaming response
            stream = client.chat.completions.create(
                model=modelname,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    full_response += content
                    print(content, end='', flush=True)
            return full_response
        else:
            # Handle non-streaming response
            chat_completion = client.chat.completions.create(
                model=modelname,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False
            )
            
            assistant_message = chat_completion.choices[0].message.content
            return assistant_message
    except Exception as e:
        print("Error:", str(e))
        return None

# Test script
if __name__ == "__main__":
    # Test case 1: Basic question
    print("Test Case 1: Basic question")
    response = ask_LLM(
        modelname="llama-3.1-70b-versatile",
        systemprompt="You are a helpful assistant.",
        content="What is the capital of France?",
        max_tokens=50
    )
    print(f"Response: {response}\n")
    
# ------------------ Alternative Implementations ------------------
'''
# ------------------ Hyperlapse API Implementation ------------------
import requests
import json

API_KEY = "hypr-lab-xxxxxxxxxx"  # Replace with your Hyperlapse API key
API_BASE = "https://api.hyperlapse.ai"  # Replace with the correct base URL

def ask_LLM(modelname, systemprompt, content, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.1, presence_penalty=1.1, streaming=True):
    """
    Send a request to the Hyperlapse API and return the LLM's response.
    
    Parameters are the same as the Groq implementation.
    """
    print(modelname)
    
    # Construct the payload for the API request
    data = {
        "model": modelname,
        "messages": [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": content}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": streaming
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Send a POST request to the API
    response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, stream=streaming)
    
    if response.status_code == 200:
        if streaming:
            # Handle streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                    if chunk['choices'][0]['finish_reason'] is not None:
                        break
                    content = chunk['choices'][0]['delta'].get('content', '')
                    full_response += content
                    print(content, end='', flush=True)
            return full_response
        else:
            # Handle non-streaming response
            assistant_message = response.json()['choices'][0]['message']['content']
            return assistant_message
    else:
        print("Error:", response.status_code, response.text)
        return None

# ------------------ Local VLLM/Ollama Implementation ------------------
import requests
import json

def ask_LLM(modelname, systemprompt, content, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.1, presence_penalty=1.1):
    """
    Send a request to a local VLLM/Ollama server and return the LLM's response.
    
    Parameters are similar to the Groq implementation, but 'streaming' is not supported.
    """
    print(modelname)
    
    # URL of your local VLLM/Ollama server
    url = "http://213.173.96.19:8000/v1/chat/completions"
    
    # Headers for the request
    headers = {"Content-Type": "application/json"}
    
    # Construct the payload for the API request
    data = {
        "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",  # Using the model specified in the example
        "messages": [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": content}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    
    # Send a POST request to the local server
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        return assistant_message
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# ------------------ Together AI API Implementation ------------------
import requests
import json

TOGETHER_API_KEY = "xxxxxxxxxxxxx"  # Replace with your Together AI API key

def ask_LLM(modelname, systemprompt, content, TOGETHER_API_KEY=TOGETHER_API_KEY, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.05, presence_penalty=1.05):
    """
    Send a request to the Together AI API and return the LLM's response.
    
    Parameters are similar to the Groq implementation, but 'streaming' is not supported.
    An additional parameter TOGETHER_API_KEY is included for flexibility.
    """
    # Construct the payload for the API request
    data = {
        "model": modelname,
        "messages": [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": content}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Together AI API endpoint
    API_BASE = "https://api.together.xyz/v1"
    
    # Send a POST request to the API
    response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data)

    # Process and return the response
    if response.status_code == 200:
        assistant_message = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        return assistant_message
    else:
        print("Error:", response.status_code, response.text)
        return None
'''

