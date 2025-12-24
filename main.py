from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import io
from PIL import Image
from google.genai import Client
from typing import List, Optional
import uvicorn

# Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ No Gemini API key found.")

app = FastAPI(title="MediVault Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_with_documents(
    query: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    """Chat with uploaded medical documents using Gemini AI"""
    try:
        async with Client().aio as aclient:
            # Process uploaded files
            file_contents = []
            if files:
                for file in files:
                    if file.content_type and file.content_type.startswith('image/'):
                        # Handle image files
                        image_data = await file.read()
                        image = Image.open(io.BytesIO(image_data))
                        file_contents.append(image)
            
            # Create prompt for medical analysis
            if file_contents:
                prompt = f"""
                You are a medical assistant AI. Analyze the provided medical documents and answer the following question:
                
                Question: {query}
                
                Please provide a helpful, accurate response based on the medical information in the documents.
                If you cannot find relevant information, please say so clearly.
                Keep your response concise and focused on the question asked.
                """
            else:
                prompt = f"""
                You are a medical assistant AI. The user asked: {query}
                
                However, no medical documents were provided. Please let them know they need to upload medical documents first to get specific analysis.
                """
            
            # Generate response using Gemini
            contents = [prompt] + file_contents
            response = await aclient.models.generate_content(
                model='models/gemini-2.5-flash',
                contents=contents
            )
            
            return JSONResponse({
                "response": response.text or "I couldn't generate a response. Please try again.",
                "status": "success"
            })
            
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-image")
async def classify_image(
    file: UploadFile = File(...),
    categories: List[str] = Form([]),
    allow_new_categories: bool = Form(False)
):
    """
    Full pipeline:
    - classify image
    - generate context text
    - create summary for AiMemory
    - create embeddings
    """

    try:
        if not categories and not allow_new_categories:
            return JSONResponse({
                "category": "other",
                "context_text": "",
                "summary": "",
                "embedding": []
            })

        # Read file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="No file content received.")

        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_err:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {img_err}")

        async with Client().aio as aclient:

            # --- 1. CATEGORY CLASSIFICATION ---
            classification_prompt = build_classification_prompt(categories, allow_new_categories)

            class_resp = await aclient.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=[classification_prompt, image]
            )

            # safe access to optional .text
            class_text: str = (getattr(class_resp, "text", None) or "").strip()
            category = clean_text(class_text)
            if not category:
                category = "unknown"


            # --- 2. CONTEXT TEXT (DETAILED DESCRIPTION) ---
            context_prompt = """
            You are an assistant analyzing a medical document or medical-related image.
            Provide a detailed but concise description of ALL information visible in the image:
            - what type of document it is
            - what information is present
            - dates, values, lab results (if readable)
            - important text
            - medical relevance

            DO NOT invent information.
            Return only the description, no extra commentary.
            """

            context_resp = await aclient.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=[context_prompt, image]
            )

            context_text: str = (getattr(context_resp, "text", None) or "").strip()


            # --- 3. SUMMARY FOR AiMemory (Short + Plain) ---
            summary_prompt = f"""
            Summarize the following text into 2-3 lines.
            The summary must be:
            - plain English
            - short
            - no unnecessary details

            TEXT TO SUMMARIZE:
            {context_text}
            """

            summary_resp = await aclient.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=[summary_prompt]
            )

            summary_text: str = (getattr(summary_resp, "text", None) or "").strip()
            summary = clean_text(summary_text)


            # --- 4. EMBEDDINGS ---
            embedding_input = f"""
            CATEGORY: {category}
            SUMMARY: {summary}
            CONTEXT: {context_text}
            """.strip()

            # note: embed_content expects 'contents' (a list)
            emb_resp = await aclient.models.embed_content(
                model="models/text-embedding-004",
                contents=[embedding_input]
            )
            print(emb_resp)
            # embed_resp.embedding might be None in some failure modes — fallback to empty list
            embedding = []
            if hasattr(emb_resp, "embeddings") and emb_resp.embeddings:
                embedding = emb_resp.embeddings[0].values

            # FINAL RESPONSE
            return JSONResponse({
                "category": category,
                "context_text": context_text,
                "summary": summary,
                "embedding": embedding
            })

    except Exception as e:
        print(f"Error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Helpers ----------
def build_classification_prompt(categories: List[str], allow_new: bool) -> str:
    """
    Builds the prompt for category classification.
    """
    base = "You are a medical document classifier. Categorize the image."

    if categories:
        list_str = ", ".join(f'"{c}"' for c in categories)

        if allow_new:
            return f"""
            {base}

            Try to fit the image into one of these categories:
            {list_str}

            If none fit well, generate a NEW, concise category name.
            Return only the category name.
            """
        else:
            return f"""
            {base}

            You MUST choose exactly one category from:
            {list_str}

            If none match, return the single word: other
            """

    else:
        return """
        Analyze the image and return a single concise category name.
        """


def clean_text(text: Optional[str]) -> str:
    """
    Safely clean optional text and return lower-cased string.
    """
    if not text:
        return ""
    return text.replace('"', '').replace("'", "").strip().lower()

if __name__ == "__main__":
    print("🚀 Starting MediVault API server...")
    print(f"📋 API Documentation: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
