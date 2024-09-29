import azure.cognitiveservices.speech as speechsdk
import time
def text_to_speech(api_key, region, text, output_audio_path):
    # Set up the configuration for the speech client
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_synthesis_voice_name =  "de-DE-FlorianMultilingualNeural"  # "de-DE-SeraphinaMultilingualNeural"  # de-DE-FlorianMultilingualNeural

    # Initialize the speech synthesizer
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_audio_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize the speech
    result = synthesizer.speak_text_async(text).get()

    # Check the result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

# Usage
api_key = 'xxxxxxxxxxxxxx'
region = 'germanywestcentral'  # such as 'eastus', 'westus', etc.
text = 'LAION ist eine deutsche Non-Profit-Organisation, die Open-Source-Modelle und -Datensätze für künstliche Intelligenz herstellt.'
output_audio_path = 'output.wav'
for i in range(5):
  s=time.time()
  text_to_speech(api_key, region, text, output_audio_path)

  print(time.time()-s)
