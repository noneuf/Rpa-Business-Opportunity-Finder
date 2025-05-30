from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv 
from crawler import crawl_website
from crawler_async import filter_relevant_urls_with_openai

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class CrawlRequest(BaseModel):
    url: str
    prompt: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def read_form():
    return FileResponse("static/index.html")

@app.get("/models")
async def get_models():
    try:
        model_list = client.models.list()
        gpt_models = [m.id for m in model_list.data if m.id.startswith("gpt")]
        return {"models": sorted(gpt_models)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/start")
async def start_crawl(req: Request, request: CrawlRequest):
    try:
        if await req.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        urls = crawl_website(request.url, max_depth=1)

        if await req.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        results = await filter_relevant_urls_with_openai(
            urls,
            batch_size=8,
            prompt=request.prompt,
            model=request.model,
            on_disconnect=req.is_disconnected
        )

        if await req.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if results:
            df = pd.DataFrame(results)
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            return {"status": "done", "download_url": f"/download/{filename}"}
        else:
            return {"status": "empty"}

    except HTTPException as e:
        print("⛔ Process stopped: Client disconnected.")
        return {"status": "aborted"}

@app.get("/download/{filename}")
async def download_excel(filename: str):
    filepath = os.path.join(".", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=filename)
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run("rpa_web_app:app", host="0.0.0.0", port=8000, reload=True)
