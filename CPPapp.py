
import streamlit as st
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# -------------------------------
# STREAMLIT CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="C++ AI Assistant",
    page_icon="💻",
    layout="centered"
)

st.title("💻 C++ AI Assistant")
st.caption("Ask questions based on your C++ dataset.")


# -------------------------------
# DATASET PATH
# -------------------------------
DATASET_PATH = Path(__file__).parent / "CPP.txt"


# -------------------------------
# LOAD DATASET AND CREATE SEARCH
# -------------------------------
@st.cache_resource
def create_vectorstore():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Read the uploaded text dataset
    text = DATASET_PATH.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if not text.strip():
        raise ValueError("CPP.txt is empty.")

    # Split the dataset into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    documents = splitter.create_documents([text])

    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Store chunks in a vector database
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="cpp_knowledge"
    )

    return vectorstore


# -------------------------------
# LOAD THE AI MODEL
# -------------------------------
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()


# -------------------------------
# CREATE PROMPT
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful C++ programming assistant.

        Answer questions primarily using the supplied context
        from the user's CPP.txt dataset.

        Explain concepts clearly and use simple language.
        Give examples or code when relevant.

        If the context does not contain enough information
        to answer, say that the answer is not available
        in the supplied dataset. Do not invent facts and
        pretend they came from the dataset.

        Context:
        {context}
        """
    ),
    (
        "user",
        """
        Previous conversation:
        {history}

        Question: {question}

        Answer:
        """
    )
])


# -------------------------------
# CREATE CHAT HISTORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------
# USER INPUT
# -------------------------------
question = st.chat_input("Ask a C++ question...")


# -------------------------------
# GENERATE ANSWER
# -------------------------------
if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching your dataset..."):

            vectorstore = create_vectorstore()

            # Find the most relevant dataset chunks
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # Keep recent conversation history
            history = "\n".join(
                f'{m["role"]}: {m["content"]}'
                for m in st.session_state.messages[-6:-1]
            )

            # Build the LangChain pipeline
            chain = prompt | llm | output_parser

            answer = chain.invoke({
                "context": context,
                "history": history,
                "question": question
            })

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        st.error(
            "Something went wrong. Check that Ollama is "
            "running, the required models are installed, "
            "and CPP.txt is in the project folder."
        )
        st.caption(str(e))


# -------------------------------
# CLEAR CHAT
# -------------------------------
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()