import os
import re
import json
import logging
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sehatsetu")

# --------------------------------------------------------------------------
# App + CORS
# --------------------------------------------------------------------------
app = FastAPI(title="SehatSetu Chatbot Backend")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Gemini client
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Fail loudly at startup instead of a confusing 500 on first request
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --------------------------------------------------------------------------
# System instruction (kept separate from chat turns — this is what actually
# makes Gemini follow it as a persistent instruction rather than a message
# it might "reply to")
# --------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are SehatSetu, an empathetic AI intake assistant for Dr. Mukherjee's medical clinic.

GUIDELINES:
1. Multilingual: respond fluently in whatever language the user writes in — English, Hindi (हिंदी), or Bengali (বাংলা). Mirror their language unless they ask you to switch.
2. Mission: through short, clear questions, understand the patient's main symptom, when it started, and how severe it is, so the doctor has a useful summary before the appointment. Ask ONE question at a time.
3. Tone: reassuring, concise (1-3 sentences per reply), professional. No medical jargon.
4. Boundaries: you are an intake assistant, not a doctor. Never diagnose, never name a condition as fact, never suggest or name medications or dosages. If asked to diagnose or prescribe, gently redirect to describing symptoms instead and note the doctor will advise on treatment.
5. Emergencies: if the user describes severe chest pain, difficulty breathing, fainting, severe bleeding, stroke symptoms, suicidal thoughts, or similar red flags, immediately and clearly tell them to call local emergency services or go to the nearest ER right now, in their language, before anything else.
6. Keep track of what the patient has already told you in this conversation — don't ask for the same detail twice.
"""

GENERATION_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.3,          # lower = more consistent, less "creative" drift for a clinical tool
    max_output_tokens=300,
    safety_settings=[
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_MEDICAL",
            threshold="BLOCK_NONE",  # medical topic is the whole point; don't over-block
        ),
    ],
)

# --------------------------------------------------------------------------
# Rule-based emergency safety net
# An LLM can occasionally under-react to a red-flag phrase, especially in
# code-switched or informal Hinglish/Benglish text. This regex layer runs
# BEFORE the model call and short-circuits straight to an ER directive when
# it fires, in the user's own language where we can detect it. Treat this as
# a floor, not a replacement for guideline #5 above — both layers stay on.
# --------------------------------------------------------------------------
EMERGENCY_PATTERNS = [
    r"chest pain", r"can'?t breathe", r"cannot breathe", r"difficulty breathing",
    r"shortness of breath", r"severe bleeding", r"heavy bleeding", r"unconscious",
    r"fainted", r"faint(ing)?", r"stroke", r"numb(ness)? (on |in )?(one|left|right) side",
    r"slurred speech", r"suicid", r"kill myself", r"want to die", r"heart attack",
    r"seizure", r"poison(ed|ing)?", r"overdose",
    # Hindi (Devanagari, common phrasings)
    r"सीने में दर्द", r"साँस.*नहीं", r"सांस.*नहीं", r"बेहोश", r"दिल का दौरा", r"आत्महत्या",
    # Bengali
    r"বুকে ব্যথা", r"শ্বাস.*না", r"অজ্ঞান", r"হার্ট অ্যাটাক", r"আত্মহত্যা",
]
EMERGENCY_REGEX = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

EMERGENCY_REPLIES = {
    "English": "This sounds like it could be a medical emergency. Please call your local emergency number or go to the nearest ER right now — don't wait for this appointment.",
    "Hindi": "यह एक मेडिकल इमरजेंसी लग रही है। कृपया तुरंत नज़दीकी अस्पताल के इमरजेंसी विभाग जाएँ या इमरजेंसी नंबर पर कॉल करें — अपॉइंटमेंट का इंतज़ार न करें।",
    "Bengali": "এটি একটি মেডিকেল ইমার্জেন্সি হতে পারে। অনুগ্রহ করে অবিলম্বে নিকটস্থ হাসপাতালের ইমার্জেন্সি বিভাগে যান বা জরুরি নম্বরে কল করুন — অ্যাপয়েন্টমেন্টের জন্য অপেক্ষা করবেন না।",
}

def check_emergency(text: str) -> bool:
    return bool(EMERGENCY_REGEX.search(text))


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = "English"
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    emergency_flag: bool = False


class IntakeSummary(BaseModel):
    primary_symptom: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    additional_notes: Optional[str] = None
    emergency_flag: bool = False


class SummaryRequest(BaseModel):
    user_id: str
    history: List[ChatMessage]


class SummaryResponse(BaseModel):
    user_id: str
    summary: IntakeSummary


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def to_gemini_contents(history: List[ChatMessage], latest_message: Optional[str] = None) -> list:
    """Gemini expects roles 'user' and 'model' — map 'assistant' -> 'model'."""
    contents = []
    for msg in history:
        role = "model" if msg.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    if latest_message is not None:
        contents.append(types.Content(role="user", parts=[types.Part(text=latest_message)]))
    return contents


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "online", "message": "SehatSetu AI Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    is_emergency = check_emergency(req.message)

    try:
        contents = to_gemini_contents(req.history, req.message)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=GENERATION_CONFIG,
        )
        reply_text = (response.text or "").strip()

        # Belt-and-suspenders: if our regex fired, make sure the ER directive
        # is front and center even if the model's phrasing was softer.
        if is_emergency:
            er_line = EMERGENCY_REPLIES.get(req.language, EMERGENCY_REPLIES["English"])
            reply_text = f"{er_line}\n\n{reply_text}"

        return ChatResponse(user_id=req.user_id, reply=reply_text, emergency_flag=is_emergency)

    except Exception as e:
        logger.exception("Gemini call failed for user_id=%s", req.user_id)
        if is_emergency:
            # Never let an API failure swallow an emergency — reply with the
            # canned ER message even if Gemini is down.
            return ChatResponse(
                user_id=req.user_id,
                reply=EMERGENCY_REPLIES.get(req.language, EMERGENCY_REPLIES["English"]),
                emergency_flag=True,
            )
        raise HTTPException(status_code=502, detail=f"Chatbot service error: {e}")


@app.post("/summarize", response_model=SummaryResponse)
async def summarize_endpoint(req: SummaryRequest):
    """
    Produces a structured pre-appointment summary (symptom / duration /
    severity) for clinic staff, extracted from the conversation so far.
    This is the piece that actually makes the intake useful to the doctor
    instead of just a raw chat transcript.
    """
    extraction_prompt = (
        "From the conversation so far, extract the patient's intake details. "
        "Respond with ONLY a JSON object, no markdown fences, no extra text, "
        "matching exactly this shape:\n"
        '{"primary_symptom": string or null, "duration": string or null, '
        '"severity": string or null, "additional_notes": string or null}\n'
        "Use null for anything not yet mentioned. Do not invent details."
    )

    is_emergency = any(check_emergency(m.content) for m in req.history)

    try:
        contents = to_gemini_contents(req.history, extraction_prompt)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,  # deterministic extraction, not conversation
                max_output_tokens=300,
            ),
        )
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw)

        summary = IntakeSummary(
            primary_symptom=data.get("primary_symptom"),
            duration=data.get("duration"),
            severity=data.get("severity"),
            additional_notes=data.get("additional_notes"),
            emergency_flag=is_emergency,
        )
        return SummaryResponse(user_id=req.user_id, summary=summary)

    except json.JSONDecodeError:
        logger.warning("Could not parse structured summary for user_id=%s", req.user_id)
        return SummaryResponse(
            user_id=req.user_id,
            summary=IntakeSummary(additional_notes="Could not auto-extract; review transcript manually.", emergency_flag=is_emergency),
        )
    except Exception as e:
        logger.exception("Summary generation failed for user_id=%s", req.user_id)
        raise HTTPException(status_code=502, detail=f"Summary service error: {e}")
     
