import os
import glob
import base64
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph

# 1) Load environment/config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Set it and restart your terminal.")

base_dir = os.path.dirname(__file__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Output folders for generated media
output_dir = os.path.join(base_dir, "outputs")
image_output_dir = os.path.join(output_dir, "images")
video_output_dir = os.path.join(output_dir, "videos")
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)

# 2) Prepare documents (load from ./data/*.txt or fall back to a default doc)
data_dir = os.path.join(base_dir, "data")
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

documents = []
if txt_files:
    for path in txt_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(page_content=text, metadata={"source": path}))
        except Exception as e:
            print(f"Skipping {path}: {e}")
else:
    documents = [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) combines vector search with language models to provide more accurate answers using external knowledge sources.",
            metadata={"source": "sample"},
        )
    ]

# 3) Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

# 4) Build vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(splits, embeddings)

# 5) Create multiple LLMs
llm_gpt4 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define state for the graph
class QueryState(TypedDict):
    question: str
    context: str
    model_choice: str
    task: str
    response: str
    media_path: str

# 6) Prompt template
prompt = """
Use only the context provided to answer the following question.
If you don't know the answer, reply that you are unsure.
Context: {context}
Question: {question}
""".strip()

prompt_template = ChatPromptTemplate.from_template(prompt)

def _extract_prompt(question: str) -> str:
    if ":" in question:
        return question.split(":", 1)[1].strip()
    return question.strip()

# Define nodes
def route_model(state: QueryState) -> QueryState:
    """
    Routing:
    - image: <prompt> -> image generation
    - video: <prompt> -> video generation
    - otherwise -> QA RAG flow
    """
    q = state["question"].strip()
    low = q.lower()

    if low.startswith("image:"):
        state["task"] = "image"
        state["model_choice"] = ""
        return state

    if low.startswith("video:"):
        state["task"] = "video"
        state["model_choice"] = ""
        return state

    state["task"] = "qa"
    state["model_choice"] = "gpt-4" if len(q) > 50 else "gpt-3.5"
    return state

def retrieve_context(state: QueryState) -> QueryState:
    """Retrieve relevant context for QA only."""
    if state["task"] != "qa":
        state["context"] = ""
        return state

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    docs = retriever.invoke(state["question"])
    state["context"] = "\n".join([doc.page_content for doc in docs])
    return state

def answer_gpt4(state: QueryState) -> QueryState:
    """Answer using GPT-4"""
    chain = prompt_template | llm_gpt4 | StrOutputParser()
    state["response"] = chain.invoke({"context": state["context"], "question": state["question"]})
    return state

def answer_gpt35(state: QueryState) -> QueryState:
    """Answer using GPT-3.5"""
    chain = prompt_template | llm_gpt35 | StrOutputParser()
    state["response"] = chain.invoke({"context": state["context"], "question": state["question"]})
    return state

def generate_image(state: QueryState) -> QueryState:
    """Generate image and save to outputs/images."""
    prompt_text = _extract_prompt(state["question"])
    if not prompt_text:
        state["response"] = "Please provide a prompt after 'image:'."
        return state

    image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    try:
        result = openai_client.images.generate(
            model=image_model,
            prompt=prompt_text,
            size="1024x1024",
        )
        image_b64 = result.data[0].b64_json
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(image_output_dir, filename)

        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        state["media_path"] = file_path
        state["response"] = f"Image generated: {file_path}"
    except Exception as e:
        state["response"] = f"Image generation failed: {e}"

    return state

def generate_video(state: QueryState) -> QueryState:
    """Generate video and save to outputs/videos."""
    prompt_text = _extract_prompt(state["question"])
    if not prompt_text:
        state["response"] = "Please provide a prompt after 'video:'."
        return state

    if not hasattr(openai_client, "videos") or not hasattr(openai_client.videos, "generate"):
        state["response"] = (
            "Video generation is not available in this OpenAI SDK version/account. "
            "Upgrade SDK and verify model access."
        )
        return state

    video_model = os.getenv("OPENAI_VIDEO_MODEL", "sora")
    try:
        result = openai_client.videos.generate(
            model=video_model,
            prompt=prompt_text,
            size="1280x720",
        )

        video_b64 = None
        if hasattr(result, "data") and result.data:
            video_b64 = getattr(result.data[0], "b64_json", None)

        if not video_b64:
            state["response"] = (
                "Video request submitted, but no downloadable payload returned. "
                "Check model/access."
            )
            return state

        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        file_path = os.path.join(video_output_dir, filename)

        with open(file_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        state["media_path"] = file_path
        state["response"] = f"Video generated: {file_path}"
    except Exception as e:
        state["response"] = f"Video generation failed: {e}"

    return state

# Build the graph
graph_builder = StateGraph(QueryState)
graph_builder.add_node("route", route_model)
graph_builder.add_node("retrieve", retrieve_context)
graph_builder.add_node("gpt4", answer_gpt4)
graph_builder.add_node("gpt35", answer_gpt35)
graph_builder.add_node("image", generate_image)
graph_builder.add_node("video", generate_video)

graph_builder.set_entry_point("route")
graph_builder.add_edge("route", "retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    lambda state: state["task"] if state["task"] in {"image", "video"} else state["model_choice"],
    {
        "gpt-4": "gpt4",
        "gpt-3.5": "gpt35",
        "image": "image",
        "video": "video",
    },
)

graph_builder.set_finish_point("gpt4")
graph_builder.set_finish_point("gpt35")
graph_builder.set_finish_point("image")
graph_builder.set_finish_point("video")

# Compile and run
graph = graph_builder.compile()

user_input = input("Ask a question, or use image: ... / video: ...\n> ")
result = graph.invoke(
    {
        "question": user_input,
        "context": "",
        "model_choice": "",
        "task": "",
        "response": "",
        "media_path": "",
    }
)

print(result["response"])
