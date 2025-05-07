from google.cloud import speech_v1
from google.cloud import texttospeech

def test_credentials():
    try:
        # Test Speech-to-Text client
        speech_client = speech_v1.SpeechClient()
        print("✅ Successfully initialized Speech-to-Text client")
        
        # Test Text-to-Speech client
        tts_client = texttospeech.TextToSpeechClient()
        print("✅ Successfully initialized Text-to-Speech client")
        
        print("\nAll Google Cloud clients initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing Google Cloud clients: {str(e)}")
        return False

if __name__ == "__main__":
    test_credentials() 