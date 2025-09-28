#!/bin/bash

# ask verbally using text-to-speech
espeak "Please say the number of pets you have after the beep"

# Play a beep to signal recording start
echo -e "\a"
sleep 1

# Recording 5 seconds of audio from microphone
echo "Recording..."
arecord --format=cd --duration=5 --file-type=wav answer.wav
echo "Recording complete.

# Transcribe the audio using Whisper CLI
echo "Transcribing your response..."
whisper answer.wav --model tiny --output_format txt --output_dir ./ > /dev/null

if [-f "answer.txt"]; then
	echo "You said: $(cat answer.txt)"

else
	echo "No transcription result found."
fi

