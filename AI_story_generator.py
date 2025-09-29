import requests
import os

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")


def generate_story(keywords):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{
            "parts": [{
                "text": f"Write a short, simple children's story using these words: {keywords}. "
                        f"Make it fun, easy to read, and end with a good moral."
            }]
        }]
    }

    response = requests.post(url, headers=headers, params=params, json=data)

    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.text}"


# 🌟 Simple User Input Loop
print("✨ AI Story Generator for Kids ✨")
print("Type a few words (like 'cat, space, friendship') and I'll create a story!\n")

while True:
    keywords = input("Enter some words (or type 'quit' to stop): ")
    if keywords.lower() == "quit":
        break
    story = generate_story(keywords)
    print("\n📖 Your Story:\n")
    print(story)
    print("\n" + "="*50 + "\n")
