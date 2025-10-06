# Chatterboxes
**NAMES OF COLLABORATORS HERE**

Jessica Hsiao(dh779), Irene Wu (yw2785)

In this lab, we designed interaction with a speech-enabled device--something that listens and talks to users. First, we storyboard what we imagine the conversational interaction to be like. Then, we will use wizarding techniques to elicit examples of what people might say, ask, or respond.  Then, we use the examples collected from at least two other people to inform the redesign of the device.

We will focus on **audio** as the main modality for interaction to start; these general techniques can be extended to **video**, **haptics** or other interactive mechanisms in the second part of the Lab.

## Part 1.

### Text to Speech

\*\***My own shell file to use my favorite of these TTS engines to have my Pi greet you by name.**\*\*

File: Lab 3/speech-scripts/greet_me.sh

https://github.com/Irene-Wu-1002/Interactive-Lab-Hub/blob/Fall2025/Lab%203/speech-scripts/greet_me.sh
  
### Speech to Text

\*\***Write my own shell file that verbally asks for a numerical based input (such as a phone number, zipcode, number of pets, etc) and records the answer the respondent provides.**\*\*

Ask number of pets

File: Lab 3/speech-scripts/ask_number.sh

https://github.com/Irene-Wu-1002/Interactive-Lab-Hub/blob/Fall2025/Lab%203/speech-scripts/ask_number.sh

\*\***Try creating a simple voice interaction that combines speech recognition, Ollama processing, and text-to-speech output. Document what I built and how users responded to it.**\*\*

file: Lab 3/ollama_voice_demo.py

https://github.com/Irene-Wu-1002/Interactive-Lab-Hub/blob/bf868fca763b423f1a7d349c37861aea52d6bf8b/Lab%203/ollama_voice_demo.py


### Storyboard

Storyboard and/or use a Verplank diagram to design a speech-enabled device. (Stuck? Make a device that talks for dogs. If that is too stupid, find an application that is better than that.) 

**Verplank diagram**
![img](https://github.com/Irene-Wu-1002/Interactive-Lab-Hub/blob/Fall2025/Lab%203/images/FairyMate.png)


**Idea**: Nowadays, individuals are more prone to mental health challenges, not only because they are constantly exposed to overwhelming amounts of information, but also because modern life often brings people physically closer yet emotionally distant. To address this, we need a way for individuals to express their feelings, whether positive or negative.

Our solution is a device called "Fairy Mate" — a friendly companion that allows individuals to share whatever is on their minds.

**Scenario**
1. When an individual returns home, Fairy Mate greets them and asks, “How was your day?”
2. The individual can freely share their thoughts and emotions.
3. Fairy Mate provides emotional support and, if possible, practical suggestions.
4. After the conversation, Fairy Mate creates a journal entry to help preserve the memory and track emotional patterns over time.

**Function**
1. Emotion Detection – Detects movement, environmental sounds, and biological signals to sense the user’s emotional state when they are at home.
2. Emotional Support – Offers comfort, encouragement, and solutions when needed.
3. Personal Journaling – Automatically creates a journal entry summarizing the user’s feelings and experiences for future reflection.


### Acting out the dialogue

Video: https://drive.google.com/file/d/1qbs6NWLszIRgVU5eu93dBdbvuezS9Y7g/view?usp=drive_link

**Difference between imagined and acted out**

Although I hadn’t explained to my friend how to use it beforehand, the conversation flowed naturally, and we were able to share real emotions during the exchange. I think it worked well because the Fairy Mate had good prompts that helped guide the discussion and made it feel caring and supportive. Even without much preparation, the interaction felt meaningful and genuine.

# Lab 3 Part 2

## Prep for Part 2

1. What are concrete things that could use improvement in the design of your device? For example: wording, timing, anticipation of misunderstandings...

- Wording: We think instead of always offering suggestions or giving emotional support, Fairy Mate could first confirm the user’s emotional tone.
- Timing:
  - Fairy Mate could pause a bit longer after a person speaks, allowing natural reflection before responding.
  - It might also recognize when the user wants to end the conversation and wrap up gently.


2. What are other modes of interaction _beyond speech_ that you might also use to clarify how to interact?

To make Fairy Mate more intuitive and engaging, we incorporated additional interaction modes beyond speech. These include touch-based controls and visual feedback, which help users understand how to interact with the device and choose the kind of support they need.

**Mode Division**

The interaction is divided into two main modes:

- Emotional Support Mode
- Solution Support Mode

These modes allow users to choose between receiving empathetic emotional comfort or practical problem-solving guidance, depending on what they need at the moment.

**Touch Interaction**

We use the Adafruit MPR121 capacitive touch sensor to let users navigate between modes by tapping numbered touch pads made from copper foil.

- Pads 1–5: Emotional Support: Provides empathetic, comforting responses. Fairy Mate listens and validates the user’s emotions without rushing to offer solutions. Example: “I hear you. That sounds really tough. Would you like to take a deep breath together?”
- Pads 6–10: Solution Support: Offers practical suggestions, motivation, or mindfulness techniques to help manage stress. Example: “Let’s think of one small thing you could do right now to feel better. Maybe take a short walk?”

**Visual Interaction**

To clarify which mode is active, Fairy Mate uses color-coded visual feedback on its screen or ambient lighting:

- Blue Light = Emotional Support Mode
- Green Light = Solution Support Mode

This visual cue helps users instantly recognize the device’s current mode and ensures smoother, more intuitive interaction — even without spoken instructions.


3. Make a new storyboard, diagram and/or script based on these reflections.

![image](https://github.com/Irene-Wu-1002/Interactive-Lab-Hub/blob/88dc1f0158ca6c594c530d43e8107a38beb3d578/Lab%203/images/Storyboard_New.png)

## Demo

[Demo Video](https://drive.google.com/drive/u/0/folders/1I0tN1MeywTAgg3vLXSNYWh-553YHKBoO)

## Test the system & Questions

### What worked well about the system and what didn't?

1. The color-coded modes (blue for Emotional Support and green for Solution Support) made it easy to recognize the device’s state.
2. The device could provide more personalized feedback based on tone or emotional cues, rather than relying only on preset responses.
3. Adding clearer visual or sound cues for when the device is “listening” versus “processing” would also make the interaction feel more intuitive.


### What worked well about the controller and what didn't?

1. The touch sensor controller worked well in providing a simple and intuitive way for users to interact with Fairy Mate.
2. The use of color feedback, blue for Emotional Support and green for Solution Support, clearly indicated which mode was active, helping users feel confident that their input was recognized.
3. The sensitivity of the touch sensor sometimes caused accidental activations or missed touches, especially if the user’s finger didn’t make full contact with the pad.
4. Because the pads were numbered rather than labeled with words or icons, some users had to remember which numbers corresponded to each mode.


### What lessons can you take away from the WoZ interactions for designing a more autonomous version of the system?

To train a more autonomous version, Fairy Mate could record anonymized data from real user interactions, such as voice input, selected touch modes, response timing, and user reactions (e.g., facial expressions, tone of voice, or posture changes). This dataset could be labeled by emotional state (e.g., sad, stressed, calm) and type of support chosen (emotional vs. solution). By analyzing patterns in how users respond to certain prompts or tones, the system could learn to adjust its responses dynamically.

### How could you use your system to create a dataset of interaction? What other sensing modalities would make sense to capture?

To train a more autonomous version, Fairy Mate could record anonymized data from real user interactions, such as voice input, selected touch modes, response timing, and user reactions (e.g., facial expressions, tone of voice, or posture changes). This dataset could be labeled by emotional state (e.g., sad, stressed, calm) and type of support chosen (emotional vs. solution). By analyzing patterns in how users respond to certain prompts or tones, the system could learn to adjust its responses dynamically.

To make Fairy Mate more emotionally intelligent, the following sensing modalities would be valuable:

**Microphone Audio Analysis**: Detect vocal tone, stress, or hesitation to infer emotional state.

**Facial Expression Recognitio**n: Use a camera to sense mood through smiles, frowns, or eye contact.

**Body Motion Sensors (Accelerometer / IMU)**: Identify restlessness, fatigue, or relaxation through movement.

**Environmental Sensors**: Measure lighting or sound levels to adjust tone or suggest activities (e.g., “It’s quiet and dark — maybe it’s time to rest”).

**Touch Pressure or Duration**: Understand emotional intensity from how long or firmly a user touches the sensor.













