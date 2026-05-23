# IMPORTS
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

####################################################################
# LOAD ENV VARIABLES
####################################################################

load_dotenv()

####################################################################
# EMBEDDING MODEL
####################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL")
)

####################################################################
# VECTOR DATABASE
####################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

####################################################################
# CHAT MODEL
####################################################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

####################################################################
# STREAMLIT PAGE
####################################################################

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🦜",
    layout="centered"
)

st.title("🦜 Agentic RAG Chatbot")

st.markdown(
    "Ask questions from your locally stored RAG knowledge base."
)

####################################################################
# SESSION STATE
####################################################################

if "messages" not in st.session_state:
    st.session_state.messages = []

####################################################################
# DISPLAY OLD CHAT MESSAGES
####################################################################

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)

####################################################################
# USER INPUT
####################################################################

user_question = st.chat_input(
    "Ask your question..."
)

####################################################################
# MAIN RAG PIPELINE
####################################################################

if user_question:

    ###############################################################
    # DISPLAY USER MESSAGE
    ###############################################################

    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append(
        HumanMessage(content=user_question)
    )

    ###############################################################
    # RETRIEVE RELEVANT DOCUMENTS
    ###############################################################

    retrieved_docs = vector_store.similarity_search_with_relevance_scores(
        user_question,
        k=5
    )

    ###############################################################
    # BUILD CONTEXT
    ###############################################################

    context = ""

    sources = []

    document_count = 1

    for doc, score in retrieved_docs:

        # Keep only relevant documents
        if score < 0.7:

            context += f"""
Document {document_count}:
{doc.page_content}

"""

            document_count += 1

            source = doc.metadata.get(
                "source",
                "Unknown Source"
            )

            if source not in sources:
                sources.append(source)

    ###############################################################
    # HANDLE EMPTY CONTEXT
    ###############################################################

    if context.strip() == "":

        context = "No highly relevant documents found."

    ###############################################################
    # PROMPT ENGINEERING
    ###############################################################

    prompt = f"""
You are an expert AI assistant.

Your task is to answer the user's question using ONLY the provided context.

Instructions:
- Give detailed and accurate explanations.
- Combine information from multiple documents when needed.
- Explain concepts clearly.
- Use professional but easy-to-understand language.
- Only answer from the provided context.
- If the answer is unavailable in the context, say:
"I don't know based on the available context."

CONTEXT:
{context}

QUESTION:
{user_question}

FINAL ANSWER:
"""

    ###############################################################
    # GENERATE RESPONSE
    ###############################################################

    try:

        with st.spinner("Generating response..."):

            response = llm.invoke(prompt)

            ai_message = response.content

    except Exception as e:

        ai_message = f"""
Error generating response:

{str(e)}
"""

    ###############################################################
    # ADD SOURCES
    ###############################################################

    if sources:

        ai_message += "\n\n---\n### Sources:\n"

        for source in sources:

            ai_message += f"- {source}\n"

    ###############################################################
    # DISPLAY AI MESSAGE
    ###############################################################

    with st.chat_message("assistant"):
        st.markdown(ai_message)

    ###############################################################
    # SAVE CHAT HISTORY
    ###############################################################

    st.session_state.messages.append(
        AIMessage(content=ai_message)
    )