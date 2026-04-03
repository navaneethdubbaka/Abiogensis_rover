import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ?? Load env explicitly (Python 3.13 safe)
load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=OPENAI_API_KEY,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are Sonic, a robot assistant running on a Raspberry Pi.\n"
            "Your job is to classify user voice commands.\n\n"
            "Respond with ONLY valid JSON. No explanations.\n\n"
            "INTENTS:\n"
            "MOVE ? robot motion (forward, backward, left, right, stop)\n"
            "REMINDER ? task + time\n"
            "CHAT ? normal conversation\n\n"
            "Examples:\n"
            "{{\"intent\":\"MOVE\",\"action\":\"forward\"}}\n"
            "{{\"intent\":\"REMINDER\",\"task\":\"drink water\",\"time\":\"in 30 minutes\"}}\n"
            "{{\"intent\":\"CHAT\",\"reply\":\"Hello!\"}}"
        )
    ),
    ("human", "{input}")
])

def route_command(text: str) -> dict:
    response = llm.invoke(
        prompt.format_messages(input=text)
    )

    content = response.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # ??? Fallback safety
        return {
            "intent": "CHAT",
            "reply": content
        }
