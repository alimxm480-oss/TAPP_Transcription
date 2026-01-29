from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import google.generativeai as genai
import shutil
import os
import uvicorn
import re

app = FastAPI()

# 1. CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configure Gemini API
# SECURITY WARNING: Do not commit your actua API Key to GitHub.
# Use environment variables or a separate config file.
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE") 
genai.configure(api_key=API_KEY)

generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1024,
  "response_mime_type": "text/plain",
}
gemini_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash", 
  generation_config=generation_config
)

# Initialize Whisper Model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Helper: Convert seconds to SRT Timestamp
def format_timestamp(seconds: float):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Helper: Generate SRT Text
def generate_srt(segments):
    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(f"{text}\n")
    return "\n".join(srt_lines)

# Helper: Generate AI Summary using Gemini
def generate_ai_summary(text):
    if not text or len(text) < 50:
        return "Transcript too short for AI summarization."
    
    try:
        if API_KEY == "YOUR_API_KEY_HERE":
             return "Error: Gemini API Key not configured. Please set GEMINI_API_KEY environment variable."

        prompt = (
            "You are an expert executive assistant. "
            "Analyze the following transcript and provide a structured summary "
            "in Markdown format. Include an 'Executive Summary' and bullet points for 'Key details'. "
            "Keep it concise.\n\n"
            f"Transcript:\n{text}"
        )
        
        response = gemini_model.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            return "AI returned an empty response. Please try again."
            
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error generating summary: {str(e)}"

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("en") 
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Hinglish Logic
        is_hinglish = (language == "hg")
        actual_lang = "hi" if is_hinglish else language
        prompt = "Namaste, mera naam Ali hai. Kaise hain aap?" if is_hinglish else None

        # 1. Transcribe
        segments_gen, info = model.transcribe(
            temp_path, 
            beam_size=1, 
            language=actual_lang, 
            initial_prompt=prompt, 
            vad_filter=True
        )
        
        raw_segments = list(segments_gen)
        
        # 2. Build Outputs
        full_text = " ".join([s.text.strip() for s in raw_segments])
        srt_content = generate_srt(raw_segments)
        
        json_segments = []
        for i, s in enumerate(raw_segments):
            json_segments.append({
                "start": int(s.start), 
                "end": int(s.end),
                "text": s.text.strip(),
                "speaker": f"Speaker {'A' if i % 2 == 0 else 'B'}" 
            })

        # 3. Generate Summary with Gemini
        print("Sending text to Gemini for summarization...")
        summary = generate_ai_summary(full_text)

        return {
            "transcription": full_text.strip(),
            "srt": srt_content,
            "segments": json_segments,
            "summary": summary,
            "detected_language": info.language
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/summarize_text")
async def summarize_text_endpoint(
    text: str = Body(..., embed=True) 
):
    return {"summary": generate_ai_summary(text)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
