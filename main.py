from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import io
from PIL import Image
from google.genai import Client  # Import the client
from google.genai import types
from typing import List

# Load environment variables
load_dotenv()
# The new Client() will automatically pick up GEMINI_API_KEY or GOOGLE_API_KEY
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Check for the other possible env var name
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file")


app = FastAPI(title="Medical Image Classifier API")

# --- CATEGORIES list is now removed ---


@app.post("/classify-image")
async def classify_image(
    file: UploadFile = File(...), 
    categories: List[str] = Form([]),
    allow_new_categories: bool = Form(False)
):
    """
    Takes an uploaded medical-related image and classifies it.
    
    - **file**: The image file to classify.
    - **categories**: (Optional) A list of preferred category names.
    - **allow_new_categories**: (Optional) If True, allows Gemini to create
      a new category if no provided category matches. Defaults to False.
    """

    try:
        # Handle the specific case where no categories are given AND new ones are not allowed
        if not categories and not allow_new_categories:
            return JSONResponse({"category": "other"})

        # Read and load image
        contents = await file.read()
        
        # Check if file is empty
        if not contents:
            raise HTTPException(status_code=400, detail="No file content received.")

        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_err:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {img_err}")



        async with Client().aio as aclient:
            
            base_prompt = "You are a document classification assistant.\nYour task is to categorize the given image."
            instruction = ""

            if categories:
                category_list_str = ", ".join(f'"{c}"' for c in categories)
                
                if allow_new_categories:
                    # ...and new categories ARE allowed (Logic from previous version)
                    instruction = f"""
                    Please categorize this image. Try to fit it into one of the following categories if one is a good match:
                    {category_list_str}

                    If none of those categories are a good fit, provide a new, more accurate and concise category name for the image.
                    """
                else:
                    # ...and new categories are NOT allowed (Restrictive)
                    instruction = f"""
                    Please categorize this image. You MUST choose the single best matching category from this list:
                    {category_list_str}

                    If no category from the list is a good match, you MUST return the single word "other".
                    """
            else:
                # If no categories are provided...
                # This block is only reachable if allow_new_categories is True (due to the check at the start)
                instruction = """
                Please analyze this image and provide a single, concise category name for it (e.g., "lab report", "invoice", "prescription").
                """

            prompt = f"""
            {base_prompt}
            {instruction}
            
            Return only the single best category name (no extra text, quotes, or explanation).
            """
            
            response: types.GenerateContentResponse = await aclient.models.generate_content(
                model="models/gemini-2.5-flash", 
                contents=[prompt, image]
            )

            category = response.text.strip().lower()
            category = category.replace('"', '').replace('.', '').strip()

            if not category:
                category = "unknown"

            return JSONResponse({"category": category})

    except Exception as e:
        print(f"An error occurred: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")