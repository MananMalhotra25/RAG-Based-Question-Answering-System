import os
import glob
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph
from typing import TypedDict

# 1) Load environment/config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Set it and restart your terminal.")

# 2) Prepare documents (load from ./data/*.txt or fall back to a default doc)
base_dir = os.path.dirname(__file__)
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
    # Fallback sample so the example runs even without local files
    documents = [
        Document(page_content="DataCamp's RAG course was created by Meri Nova and James Chapman!", metadata={"source": "sample"})
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
    response: str

# 6) Prompt template
prompt = """
Use the only the context provided to answer the following question. If you don't know the answer, reply that you are unsure.
Context: {context}
Question: {question}
""".strip()

prompt_template = ChatPromptTemplate.from_template(prompt)

# Define nodes
def route_model(state: QueryState) -> QueryState:
    """Route to appropriate model based on question complexity"""
    if len(state["question"]) > 50:
        state["model_choice"] = "gpt-4"
    else:
        state["model_choice"] = "gpt-3.5"
    return state

def retrieve_context(state: QueryState) -> QueryState:
    """Retrieve relevant context"""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 2})
    docs = retriever.invoke(state["question"])
    state["context"] = "\n".join([doc.page_content for doc in docs])
    return state

def answer_gpt4(state: QueryState) -> QueryState:
    """Answer using GPT-4"""
    chain = prompt_template | llm_gpt4 | StrOutputParser()
    state["response"] = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    return state

def answer_gpt35(state: QueryState) -> QueryState:
    """Answer using GPT-3.5"""
    chain = prompt_template | llm_gpt35 | StrOutputParser()
    state["response"] = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    return state

# Build the graph
graph_builder = StateGraph(QueryState)

graph_builder.add_node("route", route_model)
graph_builder.add_node("retrieve", retrieve_context)
graph_builder.add_node("gpt4", answer_gpt4)
graph_builder.add_node("gpt35", answer_gpt35)

graph_builder.set_entry_point("route")
graph_builder.add_edge("route", "retrieve")

# Conditional routing
graph_builder.add_conditional_edges(
    "retrieve",
    lambda state: state["model_choice"],
    {"gpt-4": "gpt4", "gpt-3.5": "gpt35"}
)

graph_builder.set_finish_point("gpt4")
graph_builder.set_finish_point("gpt35")

# Compile and run
graph = graph_builder.compile()

# Test the multi-model RAG
result = graph.invoke({
    "question": "Who are the authors?",
    "context": "",
    "model_choice": "",
    "response": ""
})

print(result["response"])
