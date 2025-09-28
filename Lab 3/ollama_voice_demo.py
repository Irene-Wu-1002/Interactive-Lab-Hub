import speech_recognition as sr
import requests
import pyttsx3

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"

# 1. Speech recongnition
def get_speech_input():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print ("Speak now...")
		audio = r.listen(source)
	try:
		text = r.recognize_google(audio)
		print(f"You said: {text}")
	except sr.UnknownValueError:
		print("Could not understand audio")
		return Nonw
	except sr.RequestError as e:
		print(f"Error with speech recognition: {e}")
		return Nonw

# 2. Ollama processing
def ask_ollama(prompt):
	response = requests.post(
		OLLAMA_URL,
		json={"model": MODEL, "prompt": prompt, "stream": False}
		)
	reply = response.json().get("response","")
	print(f"Ollama says: {reply})
	return reply

# 3. Text-to-Speech
def speak_text(text):
	engine = pyttsx3.init()
	engine.say(text)
	engine.runAndWait()

# Main Loop
if __name__ == "__main__":
	print("FairyMate Voice Interaction Started (CTRL+C to stop)")
	while True:
		user_text = get_speech_input()
		if not user_text:
			continue
		if "stop" in user_text.lower():
			print("Stopping.")
			break
		reply = ask_ollama(user_text)
		speak_text(reply)
