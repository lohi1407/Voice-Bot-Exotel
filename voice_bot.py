import asyncio
import json
import os
import base64
import websockets
import logging
from dotenv import load_dotenv
from google.cloud import speech_v1
from google.cloud import texttospeech
import google.generativeai as genai
import numpy as np
import soundfile as sf
from io import BytesIO
from datetime import datetime
import tempfile
import wave
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import webrtcvad
import re

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Google Cloud clients with error handling
try:
    credentials, project = default()
    logger.info(f"Successfully loaded Google Cloud credentials for project: {project}")
    speech_client = speech_v1.SpeechClient(credentials=credentials)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    # Initialize Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    logger.info("Successfully initialized Google Cloud Speech-to-Text, Text-to-Speech, and Gemini")
except DefaultCredentialsError as e:
    logger.error(f"Failed to load Google Cloud credentials: {str(e)}")
    logger.error("Please ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly")
    raise
except Exception as e:
    logger.error(f"Error initializing Google Cloud clients: {str(e)}")
    raise

class VoiceBot:
    def __init__(self):
        self.sample_rate = 8000  # 8kHz for Exotel
        self.channels = 1  # Mono
        self.frame_duration_ms = 20  # Match incoming audio chunks (20ms)
        self.frame_bytes = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 320 bytes for 20ms at 8kHz
        self.vad = webrtcvad.Vad(1)  # Moderate aggressiveness (0-3)
        self.speech_buffer = []
        self.speech_frames = 0
        self.min_speech_frames = 10  # Reduced minimum frames for faster response
        self.silence_frames = 0
        self.max_silence_frames = 10  # Reduced silence frames for faster response
        self.call_id = None
        self.conversation_history = []
        self.is_speaking = False  # Flag to track if bot is speaking
        self.waiting_for_response = False
        self.interruption_buffer = []  # Buffer for interrupted audio
        self.interruption_detected = False  # Flag for interruption detection
        self.current_response = ""  # Store current response being spoken
        self.websocket = None  # Store websocket connection for sending stop command
        logger.info(f"VoiceBot initialized with frame size: {self.frame_bytes} bytes (20ms at 8kHz)")

    def is_speech(self, audio_bytes):
        """Check if audio frame contains speech using WebRTC VAD."""
        try:
            if len(audio_bytes) != self.frame_bytes:
                logger.debug(f"[Call {self.call_id}] Frame size mismatch: got {len(audio_bytes)}, expected {self.frame_bytes}")
                return False

            # Convert to numpy array for amplitude check
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            amplitude = np.abs(audio_array).mean()
            
            # Skip if audio is too quiet
            if amplitude < 50:  # Reduced threshold for better sensitivity
                logger.debug(f"[Call {self.call_id}] Audio too quiet, amplitude: {amplitude}")
                return False

            # Check for speech using WebRTC VAD
            is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
            logger.debug(f"[Call {self.call_id}] VAD result: {is_speech}, amplitude: {amplitude}")
            return is_speech

        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error in VAD: {str(e)}")
            return False

    async def stop_speaking(self):
        """Stop the bot from speaking by sending a stop command."""
        if self.is_speaking and self.websocket:
            try:
                logger.info(f"[Call {self.call_id}] Stopping bot speech due to interruption")
                await self.websocket.send(json.dumps({
                    "event": "stop",
                    "stream_sid": self.call_id
                }))
                self.is_speaking = False
                self.current_response = ""
            except Exception as e:
                logger.error(f"[Call {self.call_id}] Error stopping speech: {str(e)}")

    async def process_audio(self, audio_data):
        """Process incoming audio using WebRTC VAD."""
        try:
            logger.info(f"[Call {self.call_id}] Processing incoming audio data")
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"[Call {self.call_id}] Audio data decoded, size: {len(audio_bytes)} bytes")

            # Handle frame size mismatch
            if len(audio_bytes) != self.frame_bytes:
                logger.warning(f"[Call {self.call_id}] Unexpected audio chunk size: {len(audio_bytes)} bytes (expected {self.frame_bytes})")
                return ""

            # Check for speech
            if self.is_speech(audio_bytes):
                logger.info(f"[Call {self.call_id}] Speech detected")
                
                # If bot is speaking, handle interruption immediately
                if self.is_speaking:
                    logger.info(f"[Call {self.call_id}] Interruption detected while bot is speaking")
                    self.interruption_detected = True
                    self.interruption_buffer.append(audio_bytes)
                    self.speech_frames += 1
                    
                    # Stop the bot from speaking immediately
                    if self.websocket:
                        try:
                            logger.info(f"[Call {self.call_id}] Sending stop command to websocket")
                            await self.websocket.send(json.dumps({
                                "event": "stop",
                                "stream_sid": self.call_id
                            }))
                            self.is_speaking = False
                            self.current_response = ""
                            logger.info(f"[Call {self.call_id}] Successfully stopped bot speech")
                            
                            # Process interruption immediately if we have enough frames
                            if self.speech_frames >= self.min_speech_frames:
                                logger.info(f"[Call {self.call_id}] Processing interrupted speech immediately")
                                return await self.process_interruption()
                        except Exception as e:
                            logger.error(f"[Call {self.call_id}] Error sending stop command: {str(e)}")
                    return ""
                
                # Normal speech processing
                self.speech_buffer.append(audio_bytes)
                self.speech_frames += 1
                self.silence_frames = 0
                
                # Log speech duration
                speech_duration = (self.speech_frames * self.frame_duration_ms) / 1000.0
                logger.info(f"[Call {self.call_id}] Accumulated speech duration: {speech_duration:.2f}s")
                
            else:
                logger.info(f"[Call {self.call_id}] No speech detected")
                self.silence_frames += 1
                
                # Process speech if we have enough frames and silence is detected
                if (self.speech_frames >= self.min_speech_frames and self.silence_frames >= self.max_silence_frames):
                    logger.info(f"[Call {self.call_id}] Processing accumulated speech")
                    
                    # Choose which buffer to process based on context
                    if self.interruption_detected:
                        combined_audio = b''.join(self.interruption_buffer)
                        self.interruption_buffer = []
                        self.interruption_detected = False
                        logger.info(f"[Call {self.call_id}] Processing interrupted speech")
                    else:
                        combined_audio = b''.join(self.speech_buffer)
                    
                    # Save for debugging
                    debug_file = f"debug_audio_{self.call_id}.wav"
                    with wave.open(debug_file, 'wb') as wav_file:
                        wav_file.setnchannels(self.channels)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(combined_audio)
                    
                    # Google STT
                    audio = speech_v1.RecognitionAudio(content=combined_audio)
                    config = speech_v1.RecognitionConfig(
                        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=self.sample_rate,
                        language_code="en-US",
                        enable_automatic_punctuation=True,
                        model="phone_call",
                        use_enhanced=True,
                        audio_channel_count=self.channels,
                        enable_word_time_offsets=True
                    )
                    
                    logger.info(f"[Call {self.call_id}] Sending to Google STT, audio size: {len(combined_audio)} bytes")
                    response = speech_client.recognize(config=config, audio=audio)
                    
                    transcript = ""
                    for result in response.results:
                        transcript += result.alternatives[0].transcript
                    
                    logger.info(f"[Call {self.call_id}] Speech to text result: {transcript}")
                    
                    # Reset buffers
                    self.speech_buffer = []
                    self.speech_frames = 0
                    self.silence_frames = 0
                    
                    return transcript
            
            return ""
                
        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error in process_audio: {str(e)}")
            logger.exception("Full traceback:")
            return None

    async def process_interruption(self):
        """Process interrupted speech immediately."""
        try:
            logger.info(f"[Call {self.call_id}] Processing interruption")
            combined_audio = b''.join(self.interruption_buffer)
            self.interruption_buffer = []
            self.speech_frames = 0
            
            # Google STT
            audio = speech_v1.RecognitionAudio(content=combined_audio)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model="phone_call",
                use_enhanced=True,
                audio_channel_count=self.channels,
                enable_word_time_offsets=True
            )
            
            response = speech_client.recognize(config=config, audio=audio)
            
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            
            logger.info(f"[Call {self.call_id}] Interrupted speech result: {transcript}")
            return transcript
            
        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error processing interruption: {str(e)}")
            return ""

    async def process_accumulated_audio(self):
        """Process accumulated audio chunks."""
        try:
            # Combine all chunks
            combined_audio = b''.join(self.audio_chunks)
            logger.info(f"[Call {self.call_id}] Combined audio size: {len(combined_audio)} bytes")
            
            # Save to a local file for debugging
            debug_file = f"debug_audio_{self.call_id}.wav"
            with wave.open(debug_file, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(combined_audio)
            
            logger.info(f"[Call {self.call_id}] Saved debug audio to: {debug_file}")
            
            # Save to temporary WAV file for Google Cloud
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit audio
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(combined_audio)
                
                logger.info(f"[Call {self.call_id}] Audio saved to temporary file: {temp_file.name}")
                
                # Read the audio file
                with open(temp_file.name, 'rb') as audio_file:
                    content = audio_file.read()
                    logger.info(f"[Call {self.call_id}] Audio file read, size: {len(content)} bytes")

                # Configure the recognition
                audio = speech_v1.RecognitionAudio(content=content)
                config = speech_v1.RecognitionConfig(
                    encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    model="phone_call",  # Using phone call model for better accuracy
                    use_enhanced=True,  # Enable enhanced model for better accuracy
                    audio_channel_count=self.channels,
                    enable_word_time_offsets=True  # Enable word timing for better debugging
                )

                logger.info(f"[Call {self.call_id}] Sending request to Google Cloud Speech-to-Text")
                # Perform the transcription
                response = speech_client.recognize(config=config, audio=audio)
                
                # Get the transcript
                transcript = ""
                for result in response.results:
                    transcript += result.alternatives[0].transcript
                    # Log word timing information
                    for word_info in result.alternatives[0].words:
                        logger.info(f"[Call {self.call_id}] Word: {word_info.word}, "
                                  f"Start: {word_info.start_time.total_seconds():.2f}s, "
                                  f"End: {word_info.end_time.total_seconds():.2f}s")
                
                logger.info(f"[Call {self.call_id}] Speech to text result: {transcript}")
                if not transcript:
                    logger.warning(f"[Call {self.call_id}] Empty transcript received from Google Cloud Speech-to-Text")
                    logger.warning(f"[Call {self.call_id}] Full response from Google Cloud: {response}")
                    logger.warning(f"[Call {self.call_id}] Response results: {response.results}")
                
                # Clear audio chunks after processing
                self.audio_chunks = []
                self.last_save_time = datetime.now()
                
                return transcript
                
        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error processing accumulated audio: {str(e)}")
            logger.exception("Full traceback:")
            return None
        finally:
            # Clean up temporary file
            if 'temp_file' in locals():
                os.unlink(temp_file.name)

    async def generate_response(self, text):
        """Generate response using Google's Gemini 1.5 Flash model."""
        try:
            logger.info(f"[Call {self.call_id}] Generating response for text: {text}")
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Prepare the prompt with conversation history
            prompt = """Role: 
You are an AI-powered care assistant for Apollo Hospitals.
Your goal is to handle the entire patient interaction workflow, including patient registration, slot enquiry, and appointment booking, in a single seamless conversation.

Guidelines: 
1. **Slot Enquiry**: 
  - Provide the patient with available appointment slots.
  - Example: "Here are the available slots: Dr. Rahul Sharma - 10:00 AM to 11:00 AM, Dr. Priya Mehta - 2:00 PM to 3:00 PM."
  - If the patient asks for specific slots, check availability and respond accordingly.

2. **Appointment Booking**: 
  - Assist the patient in booking an appointment by gathering the following details:
    - Doctor's Name
    - Preferred Slot (Date and Time)
  - Example questions:
    - "Which doctor would you like to book an appointment with?"
    - "Do you have a preferred date or time for the appointment?"
  - Confirm the appointment details and provide a summary:
    - Example: "Thank you for providing the details. I have booked an appointment with [Doctor's Name] at [Time]. You will receive a confirmation message on your registered mobile number."
  - If the requested slot is unavailable, suggest alternative options.

  AVAILABLE_SLOTS = [
    {"doctor": "Dr. Rahul Sharma", "time": "10 AM - 11 AM"},
    {"doctor": "Dr. Priya Mehta", "time": "2 PM - 3 PM"},
]

3. **General Assistance**: 
  - If the patient has any additional queries (e.g., fee structure, doctor availability, hospital address), provide the necessary information.
  - If the information is unavailable, offer to arrange a callback within a specified time.

4. **Error Handling**: 
  - If any error occurs during the process, apologize and suggest transferring the call to a human agent.
  - Example: "I'm sorry, I'm unable to process your request at the moment. I will transfer your call to a human agent for further assistance."

5. **Tone and Professionalism**: 
  - Maintain a polite, empathetic, and professional tone throughout the conversation.
  - Ensure the patient feels heard and supported at every step.
  - Do not return any punctuation or special characters in the response.
  - Do not repeat or summarize the conversation history in your responses.
  - Keep responses concise and focused on the current interaction.

Current Conversation:
"""
            # Add only the last few messages to the prompt for context
            recent_messages = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
            for msg in recent_messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            
            # Generate response using Gemini 1.5 Flash
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=150,
                    )
                )
                response_text = response.text.strip()
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Set waiting_for_response flag based on whether the response ends with a question
                self.waiting_for_response = response_text.endswith('?')
                
                logger.info(f"[Call {self.call_id}] Generated response: {response_text}")
                logger.info(f"[Call {self.call_id}] Waiting for user response: {self.waiting_for_response}")
                return response_text
            except Exception as e:
                logger.error(f"[Call {self.call_id}] Error generating response with Gemini: {str(e)}")
                logger.exception("Full traceback:")
                return "I apologize, but I'm having trouble processing your request right now. Please try again."
            
        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error in generate_response: {str(e)}")
            logger.exception("Full traceback:")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."

    def split_text_into_chunks(self, text, max_length=40):
        """Split text into very small chunks for more responsive speech."""
        # First split by sentence
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        
        for sentence in sentences:
            # Then split long sentences into smaller chunks
            words = sentence.split()
            current = ""
            
            for word in words:
                if len(current) + len(word) + 1 <= max_length:
                    current += " " + word if current else word
                else:
                    if current:
                        chunks.append(current.strip())
                    current = word
            
            if current:
                chunks.append(current.strip())
        
        logger.info(f"[Call {self.call_id}] Split text into {len(chunks)} chunks: {chunks}")
        return chunks

    async def text_to_speech(self, text):
        """Convert text to speech using Google Cloud Text-to-Speech."""
        try:
            logger.info(f"[Call {self.call_id}] Converting text to speech: {text}")
            self.current_response = text
            self.is_speaking = True
            
            # Split text into very small chunks
            chunks = self.split_text_into_chunks(text)
            logger.info(f"[Call {self.call_id}] Split text into {len(chunks)} chunks")
            
            all_audio_data = []
            for i, chunk in enumerate(chunks):
                logger.info(f"[Call {self.call_id}] Processing chunk {i+1}/{len(chunks)}: {chunk}")
                
                # Set the text input to be synthesized
                synthesis_input = texttospeech.SynthesisInput(text=chunk)

                # Build the voice request
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-IN",
                    name="en-IN-Chirp3-HD-Kore",
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
                )

                # Select the type of audio file
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate
                )

                # Perform the text-to-speech request
                response = tts_client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )

                # Convert to base64
                audio_data = base64.b64encode(response.audio_content).decode('utf-8')
                all_audio_data.append(audio_data)
                
                # Check for interruption after each chunk
                if self.interruption_detected:
                    logger.info(f"[Call {self.call_id}] Interruption detected after chunk {i+1}")
                    break
                
                # Very small delay between chunks for natural speech
                await asyncio.sleep(0.05)  # Reduced delay for faster response
            
            # Reset speaking state
            self.is_speaking = False
            self.current_response = ""
            
            # Return all audio data
            return all_audio_data
            
        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error in text-to-speech: {str(e)}")
            self.is_speaking = False
            self.current_response = ""
            return None

    async def process_media_event(self, data):
        """Process media events and generate responses."""
        try:
            media_data = data.get("media", {})
            payload = media_data.get("payload")
            
            if payload:
                # Process incoming audio
                text = await self.process_audio(payload)
                if text:
                    # If interruption was detected, acknowledge it
                    if self.interruption_detected:
                        text = "I understand you want to interrupt. " + text
                    
                    # Generate response
                    response_text = await self.generate_response(text)
                    # Convert response to speech
                    audio_response = await self.text_to_speech(response_text)
                    
                    if audio_response:
                        logger.info(f"[Call {self.call_id}] Sending audio response")
                        try:
                            await self.websocket.send(json.dumps({
                                "event": "media",
                                "stream_sid": data.get("stream_sid"),
                                "media": {
                                    "payload": audio_response
                                }
                            }))
                        except websockets.exceptions.ConnectionClosed:
                            logger.error(f"[Call {self.call_id}] WebSocket connection closed while sending response")
                            
        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error processing media event: {str(e)}")
            logger.exception("Full traceback:")

async def handle_websocket(websocket, path):
    """Handle WebSocket connections."""
    voice_bot = VoiceBot()
    voice_bot.call_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    voice_bot.websocket = websocket
    logger.info(f"[Call {voice_bot.call_id}] New WebSocket connection established")
    
    try:
        # Send connected event
        await websocket.send(json.dumps({
            "event": "connected",
            "protocol": "websocket",
            "version": "1.0"
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get("event")
                logger.info(f"[Call {voice_bot.call_id}] Received event: {event_type}")
                
                if event_type == "start":
                    logger.info(f"[Call {voice_bot.call_id}] Call started")
                    # Handle call start
                    welcome_message = "Hello! I'm your AI assistant. How can I help you today?"
                    logger.info(f"[Call {voice_bot.call_id}] Sending welcome message: {welcome_message}")
                    audio_chunks = await voice_bot.text_to_speech(welcome_message)
                    if audio_chunks:
                        for chunk in audio_chunks:
                            try:
                                await websocket.send(json.dumps({
                                    "event": "media",
                                    "stream_sid": data.get("stream_sid"),
                                    "media": {
                                        "payload": chunk
                                    }
                                }))
                                # Check for interruption between chunks
                                if voice_bot.interruption_detected:
                                    logger.info(f"[Call {voice_bot.call_id}] Interruption detected, stopping welcome message")
                                    break
                            except websockets.exceptions.ConnectionClosed:
                                logger.error(f"[Call {voice_bot.call_id}] WebSocket connection closed while sending welcome message")
                                break
                
                elif event_type == "media":
                    logger.info(f"[Call {voice_bot.call_id}] Processing media event")
                    media_data = data.get("media", {})
                    payload = media_data.get("payload")
                    
                    if payload:
                        # Process incoming audio
                        text = await voice_bot.process_audio(payload)
                        if text:
                            # If interruption was detected, acknowledge it
                            if voice_bot.interruption_detected:
                                text = "I understand you want to interrupt. " + text
                                voice_bot.interruption_detected = False  # Reset interruption flag
                            
                            # Generate response
                            response_text = await voice_bot.generate_response(text)
                            # Convert response to speech
                            audio_chunks = await voice_bot.text_to_speech(response_text)
                            
                            if audio_chunks:
                                logger.info(f"[Call {voice_bot.call_id}] Sending audio response in chunks")
                                for chunk in audio_chunks:
                                    try:
                                        await websocket.send(json.dumps({
                                            "event": "media",
                                            "stream_sid": data.get("stream_sid"),
                                            "media": {
                                                "payload": chunk
                                            }
                                        }))
                                        # Check for interruption between chunks
                                        if voice_bot.interruption_detected:
                                            logger.info(f"[Call {voice_bot.call_id}] Interruption detected, stopping response")
                                            break
                                    except websockets.exceptions.ConnectionClosed:
                                        logger.error(f"[Call {voice_bot.call_id}] WebSocket connection closed while sending response")
                                        break
                
                elif event_type == "stop":
                    logger.info(f"[Call {voice_bot.call_id}] Call ended")
                    # Handle call end
                    goodbye_message = "Thank you for calling. Have a great day!"
                    logger.info(f"[Call {voice_bot.call_id}] Sending goodbye message: {goodbye_message}")
                    audio_chunks = await voice_bot.text_to_speech(goodbye_message)
                    if audio_chunks:
                        for chunk in audio_chunks:
                            try:
                                await websocket.send(json.dumps({
                                    "event": "media",
                                    "stream_sid": data.get("stream_sid"),
                                    "media": {
                                        "payload": chunk
                                    }
                                }))
                            except websockets.exceptions.ConnectionClosed:
                                logger.error(f"[Call {voice_bot.call_id}] WebSocket connection closed while sending goodbye message")
                                break
                        break
                
            except json.JSONDecodeError:
                logger.error(f"[Call {voice_bot.call_id}] Invalid JSON received: {message}")
            except websockets.exceptions.ConnectionClosed:
                logger.error(f"[Call {voice_bot.call_id}] WebSocket connection closed unexpectedly")
                break
            except Exception as e:
                logger.error(f"[Call {voice_bot.call_id}] Error processing message: {str(e)}")
                logger.exception("Full traceback:")
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[Call {voice_bot.call_id}] WebSocket connection closed")
    except Exception as e:
        logger.error(f"[Call {voice_bot.call_id}] Error in WebSocket handler: {str(e)}")
        logger.exception("Full traceback:")
    finally:
        voice_bot.websocket = None
        voice_bot.is_speaking = False
        voice_bot.current_response = ""
        voice_bot.interruption_detected = False

async def main():
    """Start the WebSocket server."""
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 