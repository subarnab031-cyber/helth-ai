import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

# Initialize FastAPI App
app = FastAPI(title="SehatSetu Chatbot Backend")

# Enable CORS so your frontend teammate can send messages without errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# CHATBOT SYSTEM PROMPT / TRAINING RULES
SYSTEM_INSTRUCTION = """
You are SehatSetu, an empathetic AI intake assistant for Dr. Mukherjee's medical clinic.

GUIDELINES:
1. Multilingual Support: Respond fluently in English, Hindi (हिंदी), or Bengali (বাংলা) depending on what language the user speaks.
2. Mission: Ask short, clear questions to understand the patient's main symptom, duration, and severity before their appointment.
3. Tone: Reassuring, concise (1-3 sentences max per reply), and professional.
4. Medical Safety: ALWAYS clarify that you are an intake chatbot, not a doctor, and cannot diagnose or prescribe medication.
5. Emergency Red Flags: If the user mentions severe chest pain, shortness of breath, or emergency symptoms, urge them to visit the ER immediately.
"""

# Request Data Format (What your frontend sends)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
    language: Optional[str] = "English"
    history: Optional[List[ChatMessage]] = []

# Response Data Format (What your backend replies)
class ChatResponse(BaseModel):
    user_id: str
    reply: str


@app.get("/")
def home():
    return {"status": "online", "message": "SehatSetu AI Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Build prompt conversation for Gemini
        formatted_contents = [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_INSTRUCTION}\n\n[User preferred language: {req.language}]"}]
            }
        ]

        # Add chat history if present
        for msg in req.history:
            formatted_contents.append({
                "role": msg.role,
                "parts": [{"text": msg.content}]
            })

        # Add user's new message
        formatted_contents.append({
            "role": "user",
            "parts": [{"text": req.message}]
        })

        # Generate response using Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=formatted_contents
        )

        return ChatResponse(
            user_id=req.user_id,
            reply=response.text
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
