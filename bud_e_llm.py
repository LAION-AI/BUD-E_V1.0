
import requests
import json

API_KEY = "xxxxxxx" 

API_BASE = 'https://api.hyprlab.io/v1'
global textfortts
textfortts = ""

def ask_LLM(modelname, systemprompt, content, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.1, presence_penalty=1.1, streaming=True):
    print(modelname)
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
        "stream": streaming
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Send a POST request
    response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, stream=streaming)
    
    if response.status_code == 200:
        if streaming:
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
            assistant_message = response.json()['choices'][0]['message']['content']
            return assistant_message
    else:
        print("Error:", response.status_code, response.text)
        return None

'''
import requests
import json
import sseclient

def ask_LLM(modelname, systemprompt, content, temperature=0.7, top_p=0.9, max_tokens=400, frequency_penalty=1.1, presence_penalty=1.1, streaming=True):
    url = "http://213.173.96.19:8000/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    def create_messages(use_system_prompt=True):
        if use_system_prompt:
            return [
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": content}
            ]
        else:
            return [
                {"role": "user", "content": f"{systemprompt}\n\n{content}"}
            ]
    
    def make_request(messages):
        data = {
            "model": modelname,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": streaming
        }
        return requests.post(url, headers=headers, data=json.dumps(data), stream=streaming)
    
    # First attempt with system prompt
    response = make_request(create_messages(use_system_prompt=True))
    
    # If there's an error about system role, retry without it
    if response.status_code == 400 and "System role not supported" in response.text:
        print("System role not supported. Retrying without separate system prompt...")
        response = make_request(create_messages(use_system_prompt=False))
    
    if response.status_code == 200:
        if streaming:
            def generate():
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.data != "[DONE]":
                        try:
                            chunk = json.loads(event.data)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON: {event.data}")
            return generate()
        else:
            result = response.json()
            return result['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage:
if __name__ == "__main__":
    models = [
        #"neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        "google/gemma-2b-it"
    ]
    system_prompt = "You are a helpful assistant."
    user_content = "What is the capital of France?"
    
    for model in models:
        print(f"\nTesting with model: {model}")
        print("Streaming response:")
        stream = ask_LLM(model, system_prompt, user_content, streaming=True)
        if stream:
            for chunk in stream:
                print(chunk, end='', flush=True)
            print("\n")
        
        print("Non-streaming response:")
        response = ask_LLM(model, system_prompt, user_content, streaming=False)
        if response:
            print(response)

'''
