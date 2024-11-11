import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Initialize embeddings
hf_bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")

# Initialize Chroma clients for each document collection
client_google = chromadb.PersistentClient(path="/db/google")
client_tsla = chromadb.PersistentClient(path="/db/tsla")
client_uber = chromadb.PersistentClient(path="/db/uber")

# Set up vector stores
goog_vectorstore = Chroma(
    client=client_google,
    collection_name="google",
    embedding_function=hf_bge_embeddings,
)
tsla_vectorstore = Chroma(
    client=client_tsla,
    collection_name="tsla",
    embedding_function=hf_bge_embeddings,
)
uber_vectorstore = Chroma(
    client=client_uber,
    collection_name="uber",
    embedding_function=hf_bge_embeddings,
)

# Set up individual retrievers for each collection
retriever_goog = goog_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "include_metadata": True})
retriever_tsla = tsla_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "include_metadata": True})
retriever_uber = uber_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "include_metadata": True})

# Set up document compression pipeline
filter = EmbeddingsRedundantFilter(embeddings=hf_bge_embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])

# Initialize LLM
llm = Ollama(model="llama3.2")

# Streamlit UI
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Content Engine Chat Engine")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Initialize user input in session state
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Define a function to handle sending queries and clearing the input
def handle_send():
    user_input = st.session_state["user_input"]
    if user_input:
        # Determine which document collections to search based on query content
        list_retriever = []
        if "google" in user_input.lower():
            list_retriever.append(retriever_goog)
        if "tesla" in user_input.lower():
            list_retriever.append(retriever_tsla)
        if "uber" in user_input.lower():
            list_retriever.append(retriever_uber)

        # If no specific retriever is selected, use the LLM directly for response
        if not list_retriever:
            answer = llm(user_input)
            st.session_state["chat_history"].append({"query": user_input, "response": answer})
        else:
            # Combine retrievers and set up compression retriever
            combined_retriever = MergerRetriever(retrievers=list_retriever)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=pipeline,
                base_retriever=combined_retriever,
                search_kwargs={"k": 3, "include_metadata": True}
            )

            # Initialize QA chain with the compression retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True
            )

            # Run the query through the QA chain
            result = qa_chain({"query": user_input})
            answer = result["result"]

            # Append query and response to the chat history
            st.session_state["chat_history"].append({
                "query": user_input,
                "response": answer,
                "source_documents": result.get("source_documents", [])
            })

        # Clear the input field
        st.session_state["user_input"] = ""

# Display chat history at the top
st.markdown("### Chat History")
st.markdown(
    """
    <style>
    .user-message {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        background-color: #DCF8C6;
        color: black;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        width: fit-content;
        max-width: 70%;
    }
    .ai-message {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        background-color: #F1F0F0;
        color: black;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        width: fit-content;
        max-width: 70%;
    }
    .message-container {
        display: flex;
        flex-direction: column;
    }
    .message-container .ai-message {
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True
)

# Display the chat messages in reverse order (most recent at the bottom)
for chat in st.session_state["chat_history"]:
    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="user-message"><strong>User:</strong> {chat["query"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-message"><strong>Answer:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# User input for query at the bottom of the page, trigger on "Enter"
st.text_input("Enter your message:", key="user_input", on_change=handle_send)
