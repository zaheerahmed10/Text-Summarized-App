from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Text Summarizer",
    description="Text Summarization using Hugging Face T5 Transformer",
    version="1.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Hugging Face Model
# =========================

MODEL_NAME = "zaheer10/text-summarizer-model"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


# =========================
# Device
# =========================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()


# =========================
# Templates
# =========================

templates = Jinja2Templates(directory=".")


# =========================
# Input Schema
# =========================

class DialogueInput(BaseModel):
    dialogue: str


# =========================
# Clean Text
# =========================

def clean_data(text: str) -> str:

    text = re.sub(r"[\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    text = text.strip()

    return text


# =========================
# Summarization
# =========================

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    if not dialogue:
        return "Please enter some text."

    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        target = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        target[0],
        skip_special_tokens=True
    )

    return summary


# =========================
# API Test Route
# =========================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "message": "Text Summarizer API is running"
    }


# =========================
# Home Page
# =========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =========================
# Summarize API
# =========================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }