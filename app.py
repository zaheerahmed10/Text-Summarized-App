# fastapis

from fastapi import FastAPI, Request

from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer

import torch

import re

from fastapi.templating import Jinja2Templates  # UI

from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles


# initialize our fastapi app

app = FastAPI(
    title="Text Summarizer",
    description="Text Summarization using Hugging Face.",
    version="1.0"
)


# model and tokenizer

model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model"
)


# device
# fine tuning:

import torch

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


model.to(device)

templates = Jinja2Templates(directory=".")


# input schema for dialogue

class DialogueInput(BaseModel):

    dialogue: str


# clean function

import re

def clean_data(text):

    text = re.sub(r"[\n\r]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# summarization function

def summarize_dialogue(dialogue: str) -> str:

    # Clean dialogue

    dialogue = clean_data(dialogue)

    # Tokenize

    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)


    # Generate summary

    target = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )


    # Token IDs ko text mein convert karo

    summary = tokenizer.decode(
        target[0],
        skip_special_tokens=True
    )

    return summary


# APIs Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )