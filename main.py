from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


import streamlit as st
import os
import json
import redis




os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]= "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"

embeddings = OllamaEmbeddings(model="llama3")

VECTORSTORE_PATH = "faiss_index"
if os.path.exists(VECTORSTORE_PATH):
    db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

else:
        loader = PyPDFLoader("mgen.pdf")
        # loader = TextLoader("mgen.txt")
        docs = loader.load()

        #For large language models, it's not strictly required, but it can reduce noise and help with consistency, especially in RAG indexing.
        for doc in docs:
            doc.page_content = doc.page_content.lower()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,    # increase size
            chunk_overlap=200,  # more overlap between chunks
            separators = ["\n\n", "\n", ".", ";"] ,  # respects logical breaks)

        )

        documents=text_splitter.split_documents(docs)
        # This embeds the documents and builds the vector store
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(VECTORSTORE_PATH)

# # Set up retriever Lower k to reduce noise.
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})  # k increase you're getting more context chunks

# Define the LLM
#Low temperature ensures the model sticks closely to retrieved documents rather than inventing.
llm = OllamaLLM(model="llama3",temperature=0.1,stream=True)
# steam - This will enable streaming output, meaning tokens will be sent as they are generated (like a live typing effect), which improves perceived performance.

## Design ChatPrompt Template


prompt = ChatPromptTemplate.from_template("""
You are a technical assistant. You must answer **only** using the context provided.
Do not guess or add details not explicitly stated.
If the context does not include the answer, respond only with: "I don't know."                                     
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: 
{input}
Answer based ONLY on the context above. If the context does not contain a clear answer, say "I don't know."
""")





# Wrap the LLMChain in a StuffDocumentsChain
combine_docs_chain = create_stuff_documents_chain(
    llm = llm,
    prompt= prompt
)

# Build the RetrievalQA chain
qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

context_text = ""

def submit_feedback():
    feedback_data = {
        "question": st.session_state.input_text,
        "answer": st.session_state.last_answer,
        "feedback": st.session_state.feedback,
        "context" : context_text
    }

    if st.session_state.feedback == "Yes":
        st.write("Thank you for the feedback! We're glad the answer was correct and useful.")
    else:
        st.write("Thank you for the feedback! We'll work on improving the answers.")

    if os.path.exists("feedback_log.json"):
        with open("feedback_log.json", "r") as f:
            try:
                feedback_list = json.load(f)
            except json.JSONDecodeError:
                feedback_list = []
    else:
        feedback_list = []

    feedback_list.append(feedback_data)

    with open("feedback_log.json", "w") as f:
        json.dump(feedback_list, f, indent=2)

    st.session_state.input_text = ""  # ✅ safely reset input before re-render


# streamlit framework for ui design
st.title('Welcome to the MgenAi chatbot \U0001F916 ')

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = "Yes"

# Text input
input_text = st.text_input(
    "Are you stuck with the cleanup or stage1 ?? Enter your question:",
    key="input_text"
)

if input_text:
    cache_response = redis.get_cached_answer(input_text)  #checking in cache
    if cache_response:
        st.session_state.last_answer = cache_response
        st.write("### Answer:")
        st.write(st.session_state.last_answer)
    else:
        context_text = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(input_text)])
        response = qa_chain.invoke({
            "context": context_text,
            "input": input_text
        })
        st.session_state.last_answer = response['answer']
        redis.set_cached_answer(input_text,st.session_state.last_answer)  #saving in cahe
        st.write("### Answer:")
        st.write(st.session_state.last_answer)

    st.radio("Was this answer helpful?", ["Yes", "No"], key="feedback")
    st.button("Submit Feedback", on_click=submit_feedback)

            
   
   

