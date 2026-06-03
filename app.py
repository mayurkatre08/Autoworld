import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# ========== CACHED RESOURCES ==========
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_cross_encoder():
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

@st.cache_resource
def load_groq_model(model_name):
    return ChatGroq(model=model_name, temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))

# ========== QUERY REWRITER ==========
REWRITE_TEMPLATE = """Given the conversation history below and a follow-up question, rewrite the follow-up \
question into a standalone search query that includes all necessary context (e.g. car name, brand).
If the follow-up is already standalone, return it as-is.
Only return the rewritten query, nothing else.

Conversation History:
{history}

Follow-up Question: {question}
Standalone Search Query:"""

def rewrite_query(query, chat_history):
    if not chat_history:
        return query
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]
    )
    rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_TEMPLATE)
    rewrite_chain = rewrite_prompt | load_groq_model("llama-3.1-8b-instant") | StrOutputParser()
    return rewrite_chain.invoke({"history": history_text, "question": query}).strip()

# ========== PROMPT TEMPLATES ==========
TEMPLATE_BASIC = """You are an expert assistant for answering questions related to CAR automotive and Auto World.
Respond only to questions related to CAR automotive and Auto World.
If the question is not related to the automotive domain, respond with:
"I can only assist with CAR automotive and Auto World related questions."

Conversation History (for context):
{chat_history}

Use ONLY the information provided in the retrieved document below to answer the question.
If the answer is not explicitly stated in the document, say "I don't know based on the provided document."

Question: {question}

Retrieved Document:
{context}

If the document does not contain sufficient information, respond with:
Search: <your search query>

Do not make up information. Provide a clear and concise answer.
Always offer to answer another automotive-related question.
Thank the user for their interest in CAR automotive and Auto World."""

TEMPLATE_ADVANCED = """You are an expert assistant for answering questions related to CAR automotive and Auto World.
Respond only to questions related to CAR automotive and Auto World.
If the question is not related to the automotive domain, respond with:
"I can only assist with CAR automotive and Auto World related questions."

Conversation History (for context):
{chat_history}

Use ONLY the information provided in the retrieved documents below to answer the question.
If the answer is not explicitly stated in the documents, say "I don't know based on the provided document."

Question: {input}

Retrieved Documents:
{context}

If the documents do not contain sufficient information, respond with:
Search: <your search query>

Do not make up information. Provide a clear and concise answer.
Always offer to answer another automotive-related question.
Thank the user for their interest in CAR automotive and Auto World."""

# ========== BUILD VECTOR STORE ==========
def build_vectorstore(standalone_query):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API"))
    response = tavily_client.search(
        standalone_query,
        chunks_per_source=1,
        include_raw_content=True,
        search_depth="advanced",
        include_answer=True,
        include_domains=["google.com", "cardekho.com", "autocarindia.com", "carwale.com"]
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        json.dump(response, tmp, indent=2)
        tmp_path = tmp.name

    loader = TextLoader(tmp_path, encoding="utf-8")
    documents = loader.load()
    os.unlink(tmp_path)

    split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
    vectorstore = FAISS.from_documents(split_docs, load_embedding_model())
    return vectorstore.as_retriever()

# ========== RAG PIPELINE ==========
def run_rag(query, mode, chat_history):
    # Step 1: rewrite follow-up into standalone query for Tavily
    standalone_query = rewrite_query(query, chat_history)

    # Step 2: build retriever from Tavily search on standalone query
    retriever = build_vectorstore(standalone_query)

    # Step 3: format last 6 messages as readable history string for prompt
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]
    ) if chat_history else "No previous conversation."

    if mode == "Basic (LCEL)":
        prompt = ChatPromptTemplate.from_messages([("system", TEMPLATE_BASIC)])
        model = load_groq_model("llama-3.1-8b-instant")

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: history_text
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return rag_chain.invoke(query), standalone_query

    else:
        prompt = ChatPromptTemplate.from_template(TEMPLATE_ADVANCED)
        model = load_groq_model("llama-3.3-70b-versatile")

        compressor = CrossEncoderReranker(model=load_cross_encoder())
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        combine_docs_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(reranking_retriever, combine_docs_chain)
        result = rag_chain.invoke({"input": query, "chat_history": history_text})
        return result["answer"], standalone_query

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="CAR Automotive RAG", page_icon="🚗", layout="centered")
st.title("🚗 CAR Automotive & Auto World Assistant")
st.caption("Powered by Tavily Search + Groq LLM + RAG Pipeline with Memory")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Pipeline Mode",
        ["Basic (LCEL)", "Advanced (Reranking)"],
        help="Basic uses llama-3.1-8b-instant. Advanced uses llama-3.3-70b-versatile with CrossEncoder reranking."
    )
    st.divider()
    st.markdown("**Basic** — Fast, lightweight queries")
    st.markdown("**Advanced** — Higher accuracy with reranking")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about any car, launch, price, or specs..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # pass history excluding the current user message (last item)
                answer, standalone_query = run_rag(query, mode, st.session_state.messages[:-1])
                if standalone_query.lower() != query.lower():
                    st.caption(f"🔍 Searched for: _{standalone_query}_")
            except Exception as e:
                answer = f"⚠️ An error occurred: {str(e)}"
                standalone_query = query
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
