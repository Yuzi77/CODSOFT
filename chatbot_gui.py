import openai
import re
import random
import requests
import nltk
import tkinter as tk
from datetime import datetime
from tkinter import Scrollbar, Text
import pytz

# Setup
openai.api_key = "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"
nltk.download('punkt')

user_name = ""
asked_name = False

# Predefined responses
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!", "Nice to see you again!", "Yo!", "Howdy!"],
    "goodbye": ["Goodbye!", "See you later!", "Bye!", "Take care!", "Catch you later!"],
    "thanks": ["You're welcome!", "No problem!", "Glad I could help!", "Anytime!"],
    "help": ["How can I assist you?", "I'm here to help!", "What do you need help with?"],
    "name": ["I'm your chatbot assistant.", "I'm a virtual helper powered by GPT."],
    
    "fun_facts": [
        "Honey never spoils. 3000-year-old honey was found in Egyptian tombs!",
        "Sharks existed before trees.",
        "Octopuses have three hearts and blue blood.",
        "A day on Venus is longer than its year.",
        "Wombat poop is cube-shaped."
    ],

    "jokes": [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "How does a penguin build its house? Igloos it together!"
    ],

    "quotes": [
        "“The only way to do great work is to love what you do.” - Steve Jobs",
        "“Life is what happens when you're busy making other plans.” - John Lennon",
        "“In the middle of every difficulty lies opportunity.” - Albert Einstein",
        "“Success is not in what you have, but who you are.” - Bo Bennett",
        "“Do what you can, with what you have, where you are.” - Theodore Roosevelt"
    ],

    "riddles": [
        "What has keys but can't open locks? — A piano!",
        "I speak without a mouth and hear without ears. What am I? — An echo.",
        "What can travel around the world while staying in a corner? — A stamp.",
        "The more you take, the more you leave behind. What are they? — Footsteps.",
        "What has to be broken before you can use it? — An egg."
    ],

    "motivation": [
        "Push yourself, because no one else is going to do it for you.",
        "You are stronger than you think.",
        "Dream it. Wish it. Do it.",
        "Stay positive. Work hard. Make it happen.",
        "Don't watch the clock; do what it does. Keep going."
    ],

    "tech_facts": [
        "The first computer bug was a real insect.",
        "Over 90% of the world's currency exists only on computers.",
        "The first 1GB hard drive weighed over 500 pounds.",
        "TYPEWRITER is the longest word you can type using only the top row on a QWERTY keyboard.",
        "More people own a mobile phone than a toothbrush!"
    ],

    "default": [
        "I'm not sure how to respond to that. Want to hear a joke, quote, or riddle?",
        "Hmm, that's interesting! Would you like a fun fact or motivation?",
        "Let's keep the conversation going! Ask me for something fun, like a quote or a tech fact."
    ]
}

# GPT fallback response
def get_gpt_response(user_input): 
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Weather
def get_weather(city="Delhi"): 
    if city.lower() == "delhi":
        city = "New Delhi"
    API_key = "5770971304b8eeb3f62758d0bd288d38"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            return f"Couldn't get weather for {city}. Error: {data.get('message', 'Unknown error')}"

        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"].capitalize()
        return f"The current weather in {city.title()} is {temperature}°C with {description}."
    except Exception as e:
        return f"An error occurred while fetching the weather: {str(e)}"

# Time
def get_time_from_input(user_input):
    user_input = user_input.lower()
    if "delhi" in user_input:
        return get_time_in_timezone("Asia/Kolkata")
    for tz in pytz.all_timezones:
        if tz.lower() in user_input:
            return get_time_in_timezone(tz)
    match = re.search(r"(in|at|of)\s+([a-zA-Z\s]+)", user_input)
    if match:
        city = match.group(2).strip().replace(" ", "_").title()
        for tz in pytz.all_timezones:
            if city in tz:
                return get_time_in_timezone(tz)
    return get_time_in_timezone("Asia/Kolkata")

def get_time_in_timezone(tz_name="Asia/Kolkata"):
    try:
        tz = pytz.timezone(tz_name)
        local_time = datetime.now(tz)
        return f"The current time in {tz.zone} is {local_time.strftime('%I:%M %p')}."
    except:
        return "Sorry, I couldn't fetch the time."

# Main chatbot logic
def chatbot_response(user_input):
    global user_name, asked_name

    user_input = user_input.lower().strip()

    if not user_name:
        match = re.search(r"my name is (\w+)", user_input)
        if match:
            user_name = match.group(1).capitalize()
            asked_name = True
            return f"Nice to meet you, {user_name}! How can I help you today?"
        else:
            return "Hi! What's your name? (Say 'My name is ...')"

    if "what is my name" in user_input or "do you remember my name" in user_input:
        return f"Of course, your name is {user_name}!"

    if any(word in user_input for word in ["hi", "hello", "hey"]):
        return f"{random.choice(responses['greeting'])} {user_name}, how can I help you?"

    elif "your name" in user_input:
        return random.choice(responses["name"])

    elif "time" in user_input: # the function wont return a value for a city that is not stored in database or saved under a different name
        return get_time_from_input(user_input)

    elif "weather" in user_input: # the function wont return a value for a city that is not stored in database or saved under a different name 
        match = re.search(r"weather(?: in)? ([a-zA-Z\s]+)", user_input)
        city = match.group(1).strip() if match else "Delhi"
        return get_weather(city)

    elif any(kw in user_input for kw in ["thanks", "thank you"]):
        return random.choice(responses["thanks"])

    elif any(kw in user_input for kw in ["bye", "goodbye", "see you"]):
        return random.choice(responses["goodbye"])

    elif any(kw in user_input for kw in ["help", "assist", "support"]):
        return random.choice(responses["help"])

    elif "joke" in user_input:
        return random.choice(responses["jokes"])

    elif "fact" in user_input or "fun fact" in user_input:
        return random.choice(responses["fun_facts"])

    elif "quote" in user_input:
        return random.choice(responses["quotes"])

    elif "riddle" in user_input:
        return random.choice(responses["riddles"])

    elif "motivate" in user_input or "motivation" in user_input:
        return random.choice(responses["motivation"])

    elif "tech" in user_input:
        return random.choice(responses["tech_facts"])

    else:
        # Let GPT handle unknown queries
        return get_gpt_response(user_input)

# GUI
class chatbotGUI:
    def __init__(self, root): 
        self.root = root
        self.root.title("Chatbot")
        self.root.geometry("500x550")

        self.chat_display = Text(root, wrap=tk.WORD, width=60, height=25)
        self.chat_display.grid(row=0, column=0, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        self.scrollbar = Scrollbar(root, command=self.chat_display.yview)
        self.chat_display.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.input_field = tk.Entry(root, width=45)
        self.input_field.grid(row=1, column=0, padx=10, pady=10)
        self.input_field.bind("<Return>", lambda event: self.send_messages())

        self.send_button = tk.Button(root, text="Send", command=self.send_messages)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.input_field.focus()
        self.display_message("Chatbot: Hi there! What's your name? (Say 'My name is ...')")

    def send_messages(self):
        user_input = self.input_field.get()
        if user_input:
            self.display_message("You: " + user_input)
            response = chatbot_response(user_input)
            self.display_message("Chatbot: " + response)
            self.input_field.delete(0, tk.END)

    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    chatbot = chatbotGUI(root)
    root.mainloop()
