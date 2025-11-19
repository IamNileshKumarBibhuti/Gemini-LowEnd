from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------
# MODELS
# ---------------------------
class QueryRequest(BaseModel):
    query: str

class ImageRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    answer: str


# ---------------------------
# TEXT GENERATION
# ---------------------------
@app.post("/ask/text")
async def ask_text(request: QueryRequest) -> QueryResponse:
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-pro",
            generation_config={"temperature": 0.7, "max_output_tokens": 4096}
        )

        result = model.generate_content(request.query)
        final_text = ""

        if hasattr(result, "parts"):
            for part in result.parts:
                if getattr(part, "text", None):
                    final_text += part.text + "\n"

        elif hasattr(result, "candidates"):
            for cand in result.candidates:
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        final_text += part.text + "\n"

        final_text = final_text.strip() or "⚠️ Gemini returned no text."
        return QueryResponse(answer=final_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# IMAGE GENERATION
# ---------------------------
@app.post("/ask/image")
async def ask_image(request: ImageRequest):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            [{"role": "user", "parts": [request.prompt]}],
            generation_config={"temperature": 0.7}
        )

        images = []
        for part in response.parts:
            if hasattr(part, "inline_data"):
                images.append({
                    "mime_type": part.inline_data.mime_type,
                    "data": part.inline_data.data
                })

        if not images:
            raise Exception("❌ Gemini returned no images")

        return {"images": images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "running", "model": "gemini-2.5-pro"}
