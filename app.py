import warnings
import boto3
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from QAsystem.retreivalandgeneration import get_model, get_response_llm

# Initialize FastAPI app
app = FastAPI()

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set up Bedrock embeddings
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Route for the index page
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle form submission
@app.post("/search")
async def search(user_question: str = Form(...)):
    # Load FAISS index
    faiss_index = FAISS.load_local(
        "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Get the model
    llm = get_model()

    # Get the response from LLM
    response = get_response_llm(llm, faiss_index, user_question)

    return JSONResponse(content={"response": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)