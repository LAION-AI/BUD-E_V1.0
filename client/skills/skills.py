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

    for _ in range(3):  # Loop to display the animation multiple times
        for frame in star_frames:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console window
            print(skill_response + frame)
            time.sleep(0.5)  # Wait for half a second before showing the next frame

    return skill_response, client_session




from bud_e_captioning_with_ocr import send_image_for_captioning_and_ocr, analyze_clipboard

# KEYWORD ACTIVATED SKILL: [["capture clipboard"], ["capture the clipboard"],["get clipboard"],["get the clipboard"],  ["clipboard content"], , ["look at the clipboard"] ,  ["analyze the clipboard"] , ["check clipboard"], ["check the clipboard"],  ["Zwischenablage erfassen"], ["die Zwischenablage erfassen"], ["Zwischenablage abrufen"], ["die Zwischenablage abrufen"], ["Zwischenablageinhalt"], ["Zwischenablage ansehen"], ["Zwischenablage analysieren"],["Zwischenablage an"], ["Zwischenablage ansehen"], ["Zwischenablage anschauen"], ["Zwischenablage überprüfen"], ["die Zwischenablage überprüfen"] ]
def analyze_image (transcription_response, client_session, LMGeneratedParameters=""):
    client_id = client_session.get('client_id', 'unknown_client')
    print(f"Starting clipboard capture for client: {client_id}")


    skill_response = analyze_clipboard(client_session)

    return skill_response, client_session




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




import subprocess
import sys
import io
import logging
import tempfile
import os

def execute_client_code(codesnippet):
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.debug("Entering execute_client_code")

    logger.debug("Creating temporary file for code execution")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(codesnippet)
        temp_file_path = temp_file.name

    logger.debug(f"Temporary file created at: {temp_file_path}")

    try:
        # Execute the code using subprocess
        logger.debug("Executing code using subprocess")
        result = subprocess.run([sys.executable, temp_file_path], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        
        # Capture and store the output
        output = result.stdout

        # Log and print the output
        logger.info("Code execution completed successfully")
        logger.debug("Output of executed code:")
        logger.debug(output)
        print("Output of executed code:")
        print(output)

    except subprocess.CalledProcessError as e:
        # Handle execution errors
        error_output = f"Error executing code:\n{e.output}\n{e.stderr}"
        logger.error(f"Error during code execution: {error_output}")
        print(error_output)

    except Exception as e:
        # Handle unexpected errors
        error_output = f"Unexpected error executing code: {str(e)}"
        logger.exception(f"Unexpected error during code execution: {str(e)}")
        print(error_output)

    finally:
        # Clean up the temporary file
        logger.debug(f"Cleaning up temporary file: {temp_file_path}")
        os.unlink(temp_file_path)

    logger.debug("Exiting execute_client_code")

# LM ACTIVATED SKILL: SKILL TITLE: Open Website DESCRIPTION: This skill opens a specified website on the client's default web browser. It works across different operating systems including Windows, macOS, and Linux. USAGE INSTRUCTIONS: To activate this skill, use the following format: <open_website>https://www.example.com</open_website>. Replace 'https://www.example.com' with the URL of the website you want to open. For example, if the user asks you to open Google, you would use: <open_website>https://www.google.com</open_website>. If he asks you to open youtube, output <open_website>https://www.youtube.com</open_website> - Always use this format: <open_website> ...complete url... </open_website> 
def open_website(transcription_response, client_session, LMGeneratedParameters=""):


    import re

    def strip_special_chars(input_string):
        # Define a pattern for special characters and spaces
        pattern = r'^[\s\W]+|[\s\W]+$'
        
        # Use re.sub to replace the pattern with an empty string
        stripped_string = re.sub(pattern, '', input_string)
        
        return stripped_string


    if not LMGeneratedParameters:
        return "Error: No URL provided. Please specify a URL to open.", client_session

    url = strip_special_chars (LMGeneratedParameters)
    print(url)
    client_side_code = f"""
import platform
import webbrowser
import subprocess

def open_website(url):
    system = platform.system().lower()
    
    if system == "linux":
        try:
            subprocess.Popen(['xdg-open', url])
        except FileNotFoundError:
            webbrowser.open(url)
    elif system == "darwin":  # macOS
        try:
            subprocess.Popen(['open', url])
        except FileNotFoundError:
            webbrowser.open(url)
    elif system == "windows":
        webbrowser.open(url)
    else:
        print(f"Unsupported operating system: {{system}}")
        webbrowser.open(url)

website_url = "{url}"
open_website(website_url)
    """
    
    client_session["code_for_client_execution"] = client_side_code
    
    return f"Opening the website: {url} on the client's default web browser.", client_session

'''
# Get the result from open_website
result, client_session = open_website("bla", {}, "https://youtube.com")

# Extract the client-side code from the client_session dictionary
codesnippet = client_session.get("code_for_client_execution", "")

# Now execute the client-side code
execute_client_code(codesnippet)
'''