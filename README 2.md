# Voice Bot with Exotel Integration

This project implements an AI-powered voice bot that integrates with Exotel's cloud telephony service. The bot can handle incoming calls, process speech, and respond using text-to-speech capabilities.

## Features

- Real-time audio streaming using WebSocket
- Speech-to-text conversion
- AI-powered conversation handling
- Text-to-speech response generation
- Exotel integration for call handling

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_api_key
EXOTEL_SID=your_exotel_sid
EXOTEL_TOKEN=your_exotel_token
```

4. Start the WebSocket server:
```bash
python voice_bot.py
```

5. Use ngrok to expose the server:
```bash
ngrok http 8765
```

## Configuration

Update the Exotel webhook URL in your Exotel dashboard with the ngrok URL:
```
https://your-ngrok-url/voice
```

## Usage

1. Make a call to your Exotel number
2. The bot will answer and start the conversation
3. Speak naturally and the bot will respond accordingly

## Project Structure

- `voice_bot.py`: Main WebSocket server implementation
- `requirements.txt`: Project dependencies
- `.env`: Environment variables and credentials

## Prerequisites

- Python 3.8+
- Exotel account with Voicebot Applet access
- OpenAI API key (for AI processing)

## Development

The main components are:
- `voice_bot.py`: Main server implementation
- WebSocket handlers for different events
- Audio processing pipeline

## Security Considerations

- Always use HTTPS/WSS in production
- Implement proper authentication
- Secure your API keys
- Monitor WebSocket connections

## License

MIT License 