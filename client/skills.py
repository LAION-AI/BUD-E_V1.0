import json
import random
from PIL import Image
from PIL import ImageGrab

import threading

import io
import requests
import json
import base64
import os
import re
import subprocess
import time 

import sys


# KEYWORD ACTIVATED SKILL: [["twinkle twinkle little star"], ["twinkle, twinkle, little, star"], ["twinkle twinkle, little star"], ["twinkle, twinkle little star"] , ["Twinkle, twinkle, little star"], ["twinkle, little star"], ["twinkle little star"]]
def print_twinkling_star_server_side_execution(transcription_response, client_session, LMGeneratedParameters=""):
    # Simulated animation of a twinkling star using ASCII art

    star_frames = [
        """
             ☆ 
            ☆☆☆
           ☆☆☆☆☆
            ☆☆☆
             ☆
        """,
        """
             ✦
            ✦✦✦
           ✦✦✦✦✦
            ✦✦✦
             ✦
        """
    ]

    skill_response = "Twinkle, twinkle, little star!\n"
    updated_conversation = conversation
    updated_scratch_pad = scratch_pad

    for _ in range(3):  # Loop to display the animation multiple times
        for frame in star_frames:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console window
            print(skill_response + frame)
            time.sleep(0.5)  # Wait for half a second before showing the next frame

    return skill_response, client_session



# KEYWORD ACTIVATED SKILL: [["have a look at the clipboard"], ["look at the clipboard"], ["check the clipboard"], ["what's in the clipboard"], ["show me the clipboard"], ["clipboard content"], ["view clipboard"]]
def process_clipboard_content_client_side_execution(transcription_response, client_session, LMGeneratedParameters=""):
    import pyperclip
    import re

    def is_youtube_url(url):
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        youtube_regex_match = re.match(youtube_regex, url)
        return bool(youtube_regex_match)

    def dummy_image_caption(image_path):
        return "This is a caption for the image in the clipboard."

    def dummy_youtube_processor(url):
        return f"Processed YouTube video with URL: {url}"

    clipboard_content = pyperclip.paste()

    if clipboard_content:
        if clipboard_content.startswith(('http://', 'https://')):
            if is_youtube_url(clipboard_content):
                result = dummy_youtube_processor(clipboard_content)
            else:
                result = f"The clipboard contains a URL: {clipboard_content}"
        elif clipboard_content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            result = dummy_image_caption(clipboard_content)
        else:
            result = f"The clipboard contains the following text:\n{clipboard_content}"
        
        # Append the clipboard content to the transcription response
        updated_transcription = f"{transcription_response} [Clipboard content: {result}]"
        
        skill_response = f"I've processed the clipboard content. {result}"
    else:
        skill_response = "The clipboard is empty."
        updated_transcription = transcription_response

    # Update the client session with the new transcription
    updated_client_session = client_session.copy()
    updated_client_session['last_transcription'] = updated_transcription

    return skill_response, updated_client_session

    
# KEYWORD  ######## ACTIVATED SKILL: [["change", "voice"], ["switch", "voice"], ["alter", "voice"], ["modify", "voice"],  ["ändere", "stimme"], ["wechsle", "stimme"], ["verändere", "stimme"]]
def server_side_execution_change_tts_voice(transcription_response, client_session, LMGeneratedParameters):
    print("Entering server_side_execution_change_tts_voice function")
    
    # List of available TTS server URLs
    tts_servers = [
        "http://213.173.96.19:5001",
        "http://213.173.96.19:5004"
    ]
    
    # Get the current TTS server URL from the client session
    current_tts_url = client_session.get('TTS-Config', {}).get('TTS_SERVER_URL', '')
    print(f"Current TTS Server URL: {current_tts_url}")
    
    # Select the other TTS server URL
    new_tts_url = next(url for url in tts_servers if url != current_tts_url)
    
    print(f"New TTS Server URL: {new_tts_url}")
    
    # Update the TTS server URL in the client session
    if 'TTS-Config' not in client_session:
        client_session['TTS-Config'] = {}
    client_session['TTS-Config']['TTS_SERVER_URL'] = new_tts_url
    
    # Prepare the response
    response = f"Certainly! I've changed the voice for you. The new TTS server URL is {new_tts_url}. Let me know if you'd like me to say something with the new voice!"
    
    # Add the interaction to the conversation history
    client_session['Conversation History'] = client_session.get('Conversation History', [])
    client_session['Conversation History'].append({"role": "user", "content": transcription_response})
    client_session['Conversation History'].append({"role": "assistant", "content": response})
    
    print("Exiting server_side_execution_change_tts_voice function")
    return response, client_session



# LM ACTIVATED SKILL: SKILL TITLE: Change Voice DESCRIPTION: This skill changes the text-to-speech voice for the assistant's responses. USAGE INSTRUCTIONS: To change the voice, use the following format: <change_voice>voice_name</change_voice>. Replace 'voice_name' with one of the available voices: Stella, Stefanie, Florian, or Thorsten. For example, to change the voice to Stefanie, you would use: <change_voice>Stefanie</change_voice>. The assistant will confirm the voice change or provide an error message if an invalid voice is specified.
def server_side_execution_change_voice(user_input, client_session, params):
    voice_name = params.strip('()')
    valid_voices = ['Stella', 'Stefanie', 'Florian', 'Thorsten']
    
    if voice_name not in valid_voices:
        return f"Invalid voice. Please choose from: {', '.join(valid_voices)}.", client_session
    
    if 'TTS_Config' not in client_session:
        client_session['TTS_Config'] = {}
    
    client_session['TTS_Config']['voice'] = voice_name
    
    print(f"Voice changed to {voice_name}")
    
    return f"Voice successfully changed to {voice_name}.", client_session



'''

# LM ACTIVATED SKILL: SKILL TITLE: Deep Search and Summarize Wikipedia. DESCRIPTION: This skill performs a deep search in English Wikipedia on a specified topic and summarizes all the results found. USAGE INSTRUCTIONS: To perform a deep search and summarize, use the command with the tags <deep-wikipedia> ... </deep-wikipedia>. For example, if the user wants to find information on 'Quantum Computing', you should respond with: <deep-wikipedia>Quantum Computing</deep-wikipedia>.
def deep_search_and_summarize_wikipedia(transcription_response, client_session, LMGeneratedParameters):
    """
    This skill searches English Wikipedia for a given topic and summarizes the results.
    """

    from bud_e_llm import ask_LLM

    import wikipediaapi

    system_prompt = client_session.get('System_Prompt', '')
    llm_config = client_session.get('LLM_Config', {})

    # Wikipedia API initialization
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='en_wiki_api/1.0 (me@example.com)'  # Example User-Agent
    )

    def get_wikipedia_content(topic):
        """
        This function retrieves the content of a Wikipedia article on a given topic.
        """
        page = wiki_wiki.page(topic)
        if page.exists():
            return page.text, page.fullurl
        else:
            return "No article found.", None


    print("START")
    # Fetch the content from Wikipedia
    raw_text, source_url = get_wikipedia_content(topic)

    
    if raw_text == "No article found.":
        # ...




    # Instruction for the LLM to summarize the text
    instruction = f"Summarize the following text to 500 words with respect to what is important and provide at the end source URLs with explanations : {raw_text[:5000]}"
    
    llm_reply =ask_LLM(
            llm_config['model'],
            system_prompt,
            conversation_history_str,
            temperature=llm_config['temperature'],
            top_p=llm_config['top_p'],
            max_tokens=llm_config['max_tokens'],
            frequency_penalty=llm_config['frequency_penalty'],
            presence_penalty=llm_config['presence_penalty']
        )


    # Form the final response
    #skill_response = ...

    print(skill_response)
    return skill_response, client_session

# Here is a raw draft and skeleton for a language model activated skill that should take a question or a topic and then search in Wikipedia and take the top three results and look at the top three results and try to extract snippets from each one, like from the full raw text of the top three search results that are relevant to answer this question or that are relevant, very relevant for this topic that the user asked about. And these extractions should be like basically summarized together with the source page and with quotation marks and then and then like it should um for each result like each topic or each um like for each paragraph it quotes and extract it make up a new follow-up question that could be relevant to answer the overarching question and then perform again a wikipedia search for the each of the of the of the sub questions questions and then also take the top three results for each sub and extract what is relevant together with the source and finally take everything together and then make another final language model call to formulate one final essay or reply that is really well thought and provides sources and quotations to answer the question. So basically it makes a final report and returns this final report in a nice formatting in HTML so that I could just take this and like copy paste it into a browser like copy paste it into an HTML file and display this into a browser and I would get like the relevant extracted parts as like in quotation marks with the sources and the standard like insights or the normal text that basically present this and try to draw conclusions without quotation marks and yeah like the citations should be italic but the normal text like the draw the conclusion that presents everything should be in normal text and I want a bold a bold headline and I want and yeah, I want everything very nicely formatted.

'''