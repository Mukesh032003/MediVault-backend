from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import io
from PIL import Image
from google.genai import Client
from typing import List, Optional

# Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ No Gemini API key found.")

app = FastAPI(title="MediVault Classifier API")


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
