import requests
import json


TOGETHER_API_KEY = "xxxxx"

def ask_LLM(modelname, systemprompt, content, TOGETHER_API_KEY=TOGETHER_API_KEY, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.05, presence_penalty=1.05):
    # Construct the payload
    data = {
        "model": modelname,
        "messages": [
            {
                "role": "system",
                "content": systemprompt
            },
            {
                "role": "user",
                "content": content
            }
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

    # URL updated to match the Together API endpoint
    API_BASE = "https://api.together.xyz/v1"
    # Send a POST request
    response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data)

    # Display the response (you can format this better if needed)
    if response.status_code == 200:
        assistant_message = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        return assistant_message
    else:
        print("Error:", response.status_code, response.text)
