from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv
from langfuse import Langfuse
import os
from jinja2 import Template
from langchain_core.tools import tool
from tavily import TavilyClient
import gradio as gr
import html
from gradio.themes.glass import Glass
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import fitz
import json

# Load environment variables
load_dotenv()

# Langfuse setup
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Cosine similarity function
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Tool: RAG Retriever
@tool
def retrieve(query, n=3):
    """
    This is a Rag Model that retrieves top 3 optimal answer for user's query,
    it uses vector embedding to embed query and then goes to database to find optimal
    through cosine similarities.
    """
    query_embedding = model.encode(f"query: {query}")
    similarities = []
    for i in collection.find({}, {"_id": 0, "chunk": 1, "embedding": 1}):
        chunk = i["chunk"]  
        embedding = i["embedding"]
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return "\n\n".join([chunk for chunk, _ in similarities[:n]])

# Tool: Tavily Search
@tool
def tavily(query):
    """
    tavily is a search engine like google that fetches answers majorly in json format
     that include texts and urls.
    """
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query)
    return response

# MongoDB setup
model = SentenceTransformer("intfloat/e5-base-v2")
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["Agent"]
collection = db["Netsol document"]
collection1 = db["Chat History"]

# Chat history persistence
def save_chat_history(prompt, answer, token):
    if not token or not prompt or not answer:
        return
    collection1.insert_one({"token": token, "prompt": prompt, "Answer": answer})

def chat_history(user_token):
    if not user_token:
        return []
    history = []
    for doc in collection1.find({"token": user_token}, {"_id": 0, "prompt": 1, "Answer": 1}):
        p = doc.get("prompt", "")
        a = doc.get("Answer", "")
        history.append(f"Prompt: {p}\nAnswer: {a}")
    return history

# LLM and Tools setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [tavily, retrieve]
llm_with_tools = llm.bind_tools(tools)

prompt_template = langfuse.get_prompt("react-agent-prompt")
prompt = prompt_template.get_langchain_prompt()
new_prompt = Template(prompt).render(formatted_history=chat_history)
sys_msg = SystemMessage(content=new_prompt)

# Assistant function using streaming
def assistant(state: MessagesState):
    stream = llm_with_tools.stream([sys_msg] + state["messages"])
    for chunk in stream:
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            for call in chunk.tool_calls:
                print(f"Tool called: {call['name']} with args: {call['args']}")
        if isinstance(chunk, AIMessageChunk):
            yield {"messages": [chunk]}
        elif isinstance(chunk, AIMessage):
            yield {"messages": [chunk]}

# LangGraph definition
def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.set_finish_point("assistant")
    return builder.compile()

def extract_text_from_pdf(pdf_path):
    text_chunks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            if text:
                chunks = text.split('\n\n')
                text_chunks.extend([chunk.strip() for chunk in chunks if chunk.strip()])
    return text_chunks

def take_file_call(file):
    dataset = extract_text_from_pdf(file)
    total_chunks = len(dataset)
    documents = []

    for i, chunk in enumerate(dataset):
        embedding = model.encode(f"passage: {chunk}")
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        documents.append({"chunk": chunk, "embedding": embedding})
        percent = int((i + 1) / total_chunks * 100)
        yield percent, f"Uploading chunk {i+1}/{total_chunks}..."

    collection.insert_many(documents)
    yield 100, "File Uploaded"

graph = build_graph()

def chatbot(prompt, history, token):
    result = graph.stream({"messages": [HumanMessage(content=prompt)]})
    full_response = ""
    flag = False
    prompt_history = []
    Answer = []
    for res in result:
        for value in res.values():
            if "messages" in value:
                for msg in value["messages"]:
                    if hasattr(msg, "content"):
                        try:
                            data = json.loads(msg.content)
                            if isinstance(data, dict) and "results" in data:
                                contents = [res["content"] for res in data["results"] if "content" in res]
                                for c in contents:
                                    full_response += c + "\n\n"
                                    yield full_response
                                    flag = True
                            else:
                                full_response += msg.content
                                yield full_response
                                flag = True
                        except Exception:
                            full_response += msg.content
                            yield full_response
                            flag = True
    if not flag:
        yield "Sorry I am unable to answer this."
    if full_response.strip():
        prompt_history.append(prompt)
        Answer.append(full_response.strip())
        save_chat_history(prompt, full_response, token)

def take_file(file):
        for percent, status in take_file_call(file):
            yield percent, status

# Gradio Interface
custom_css = """
    #my_chatbot_container {
        height: 100% !important;
        overflow-y: auto; /* Add scroll if content exceeds height */
    }
    """
def create_gradio_app():
    def load_history(token):
        return html.escape("\n\n".join(chat_history(token)))
    mychatbot=gr.Chatbot(height="60vh")
    with gr.Blocks(fill_height=True,css=custom_css,theme=Glass) as demo:
        gr.Markdown("### Your Chat History")
        inp = gr.Textbox(placeholder="What is your token?",elem_id="my_chatbot_container")
        out = gr.Textbox()
        inp.change(load_history, inp, out)

        gr.Markdown("### Upload PDF to Embed")
        upload = gr.UploadButton(label="Upload PDF")
        progress_slider = gr.Slider(minimum=0, maximum=100, label="Upload Progress", interactive=False)
        progress_text = gr.Textbox(label="Status", lines=2, interactive=False)

        # Yielding both progress bar and text
        upload.upload(
            fn=take_file,
            inputs=upload,
            outputs=[progress_slider, progress_text]
        )

        gr.ChatInterface(
            fn=chatbot,
            title="chatbot",
            type="messages",
            additional_inputs=[inp],
            chatbot=mychatbot
        )
    return demo


demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch(share=True)