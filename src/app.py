import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
import re
from webChat import load_and_retrieve_docs


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_vector_store(text):
    # 	embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceHubEmbeddings()
    vector_store = Chroma.from_texts(texts=text, embedding=embeddings)
    return vector_store


def get_chunk_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_chat(vector_store):
    # 	llm = ChatOpenAI()
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature = 0.5,
        task="text-generation",
        model_kwargs={"max_length": 1024},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):

    pattern = re.compile(r"Helpful Answer: (.*?)(?=Helpful Answer:|$)", re.DOTALL)

    if not st.session_state.chat:
        st.info("Upload your documents by opening sidebar!")        
    else:
        response = st.session_state.chat({"question": user_question})
        st.session_state.chat_history.append(
            {"question": user_question, "answer": response["answer"]}
        )

        # st.write(response)        


def ChatWithUrl():
    st.header("Chat with the Webpage Content")
    url = st.text_input("Enter the Url")
    if url:
        with st.spinner("processing"):
            st.session_state.conversation_chain = load_and_retrieve_docs(url)

    user_input = st.chat_input("Ask your question")

    if user_input and st.session_state.conversation_chain:
        with st.spinner("loading"):
            response = st.session_state.conversation_chain({"question": user_input})
            st.session_state.history.append(
                {"question": user_input, "answer": response["answer"]}
            )
    else:
        if not url:
            st.info("Enter the Url")

    for chat in st.session_state.history:
        st.chat_message("assistant").write(chat["question"])
        st.chat_message("user").write(chat["answer"])


def ChatWithDocuments():

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with documents :books:")
    user_question = st.chat_input("Ask a question about your uploaded documents:")
        
        
    if user_question:
        with st.spinner("loading"):
            handle_userinput(user_question)
    else:        
        pass
        # st.stop()
        
    for chat in st.session_state.chat_history:
        st.chat_message("assistant").write(chat["question"])
        st.chat_message("user").write(chat["answer"])


def main():
    load_dotenv()
    st.set_page_config(page_title="RAG with LLM", page_icon=":books:")
    
    if "page" not in st.session_state:
        st.session_state.page = True

    if st.session_state.page:
        if "chat" not in st.session_state:
            st.session_state.chat = None    
        ChatWithDocuments()
    else:
        if "history" not in st.session_state:
            st.session_state.history = []
        ChatWithUrl()

    with st.sidebar:
        st.title("RAG with LLM")
        
        if st.button("Chat with You Documents"):
            st.session_state.page = True
            
            # chat with documents                                   
            
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_chunk_text(raw_text)
                vector_store = get_vector_store(chunks)
                st.session_state.chat = get_chat(vector_store)


        # chat with url content        
        if st.button("Chat with Url content"):
            st.session_state.page = False


if __name__ == "__main__":
    main()
