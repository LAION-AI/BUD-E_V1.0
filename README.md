# BUD-E_V1.0

<p align="center">
  <img src="https://github.com/user-attachments/assets/23a4699e-5097-4184-aef6-3ac978498df4" alt="BUD-E" width="400px">
</p>

**BUD-E (Buddy)** is an open-source voice assistant framework designed to facilitate seamless interaction with AI models and APIs. It enables the creation and integration of diverse skills for educational and research applications.

---

## Architecture Overview

BUD-E_V1.0 operates on a **client-server architecture**, allowing users to communicate with the assistant on edge devices. The main computation, however, is conducted on a server, which can either be cloud-based or on a local device equipped with a strong GPU.

<p align="center">
  <img src="https://github.com/user-attachments/assets/31e0ab92-d8fa-4793-8f5b-dc02fa47db6a" alt="BUD-E Architecture" width="400px">
</p>


BUD-E V1.0 uses a client-server architecture:

- Server: Handles main computation (speech recognition, language processing, text-to-speech, vision processing).
- Client: Manages user interactions (audio recording, playback, clipboard management).

## Components

### Server Components:
1. Automatic Speech Recognition (ASR)
2. Language Model (LLM)
3. Text-to-Speech (TTS)
4. Vision Processing (Image Captioning and OCR)

### Client Options:
1. Python Desktop Client (Windows and Linux)
2. School BUD-E Web Interface

Note: Mac OS support for the desktop client is waiting for you to build it. :)

## Installation

### Server Setup

1. Clone the repository:
   ```
   git clone https://github.com/LAION-AI/BUD-E_V1.0.git
   cd BUD-E_V1.0/server
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure components in their respective files:
   - ASR: `bud_e_transcribe.py`
   - LLM: `bud_e_llm.py`
   - TTS: `bud_e_tts.py`
   - Vision: `bud_e_captioning_with_ocr.py`

4. Start the server:
   ```
   python bud_e-server.py
   ```

### Client Setup

1. Navigate to the client directory:
   ```
   cd ../client
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the client:
   - Edit `bud_e_client.py` to set the server IP and port.
   - Obtain a Porcupine API key for wake word detection.

4. Run the client:
   ```
   python bud_e_client.py
   ```

## Skill System

BUD-E's functionality can be extended through a skill system. Skills are Python functions that can be activated in two ways:

1. Keyword Activation
2. Language Model (LM) Activation

### Skill Creation

To create a new skill:

1. Create a Python file in the `client/skills` folder.
2. Define the skill function with this structure:

```python
def skill_name(transcription_response, client_session, LMGeneratedParameters=""):
    # Skill logic
    return skill_response, client_session
```

3. Add a skill description comment above the function:

For keyword-activated skills:
```python
# KEYWORD ACTIVATED SKILL: [["keyword1"], ["keyword2", "keyword3"], ["phrase1"]]
```

For LM-activated skills:
```python
# LM ACTIVATED SKILL: SKILL TITLE: Skill Name DESCRIPTION: What the skill does. USAGE INSTRUCTIONS: How to use the skill.
```

### LM-Activated Skill Example

Here's an example of an LM-activated skill that changes the assistant's voice:

```python
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
```

This skill demonstrates how LM-activated skills work:

1. The skill description provides instructions for the language model on how to use the skill.
2. The skill expects the LM to generate a parameter enclosed in specific tags (e.g., `<change_voice>Stefanie</change_voice>`).
3. The `params` argument in the function receives the content within these tags.
4. The skill processes this input and updates the client session accordingly.

## Customization

BUD-E supports integration with various AI model providers:

- ASR: Local Whisper models or cloud services (e.g., Deepgram)
- LLM: Commercial APIs (e.g., Groq, OpenAI) or self-hosted models (e.g., VLLM, Ollama)
- TTS: Cloud services or local solutions (e.g., FishTTS, StyleTTS 2)
- Vision: Custom models or cloud APIs

Refer to the configuration files for integration examples.

## Troubleshooting

Common issues and potential solutions:

1. Dependency installation failures: Try using `conda` for problematic packages.
2. API connection errors: Verify API keys, endpoint URLs, and network connectivity.
3. Wake word detection issues: Ensure correct Porcupine API key configuration.
4. Performance issues: For local setups, ensure adequate GPU capabilities or optimize model sizes.

## Contributing

Best join our Discord community: https://discord.gg/pCPJJXP7Qx

## License

Apache 2.0

## Acknowledgements

- Porcupine for wake word detection
- Whisper for speech recognition
- FishTTS and StyleTTS 2 for text-to-speech capabilities
- Groq, Hyperlab, and other API providers for AI model access
