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
import my_redis_cache
import shutil



os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]= "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"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


embeddings = OllamaEmbeddings(model="llama3")

VECTORSTORE_PATH = "faiss_index"

#clear old FAISS index if you want to rebuild it
# if os.path.exists(VECTORSTORE_PATH):
#     shutil.rmtree(VECTORSTORE_PATH)  # Deletes the old vectorstore

# Load existing FAISS index
if os.path.exists(VECTORSTORE_PATH):
    db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

else:
    loader = PyPDFLoader("mgen_issue.pdf")
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
        "context": st.session_state.context_text
    }

    if st.session_state.feedback == "Yes" and not my_redis_cache.get_cached_answer(st.session_state.input_text):
        my_redis_cache.set_cached_answer(st.session_state.input_text, st.session_state.last_answer)

    if st.session_state.feedback == "Yes":
        st.write("Thank you for the feedback! We're glad the answer was correct and useful.")
    else:
        st.write("Thank you for the feedback! We'll work on improving the answers.")
    
    feedback_file = "feedback_log.json"
    if not os.path.exists(feedback_file):
        with open(feedback_file, "w") as f:
            json.dump([], f)
    try:
        with open(feedback_file, "r") as f:
            feedback_list = json.load(f)
    except json.JSONDecodeError:
        feedback_list = []

    feedback_list.append(feedback_data)

    with open("feedback_log.json", "w") as f:
        json.dump(feedback_list, f, indent=2)

    # ✅ Use flag to clear input on next rerun
    st.session_state.clear_input = True
    st.session_state.last_answer = ""
    st.session_state.context_text = ""
    st.session_state.feedback_submitted = True
    st.rerun()


# Streamlit UI
st.title('Welcome to the MgenAi chatbot \U0001F916 ')

# Initialize session state
for key, value in {
    "input_text": "",
    "last_answer": "",
    "feedback": "Yes",
    "context_text": "",
    "feedback_submitted": False,
    "clear_input": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Clear input before rendering text_input if flag is set
if st.session_state.clear_input:
    st.session_state.input_text = ""
    st.session_state.clear_input = False

with st.form("question_form"):
    input_text = st.text_input(
        "Are you stuck with the cleanup or stage1? Enter your question:",
        key="input_text"
    )
    submitted = st.form_submit_button("Submit")

if submitted and st.session_state.input_text.strip():
    st.session_state.feedback_submitted = False

    input_text = st.session_state.input_text
    cache_response = my_redis_cache.get_cached_answer(input_text)

    if cache_response:
        st.session_state.last_answer = cache_response
    else:
        st.session_state.context_text = "\n\n".join(
            [doc.page_content for doc in retriever.invoke(input_text)]
        )
        response = qa_chain.invoke({
            "context": st.session_state.context_text,
            "input": input_text
        })
        st.session_state.last_answer = response['answer']


if st.session_state.last_answer and not st.session_state.feedback_submitted:
    st.write("### Answer:")
    st.write(st.session_state.last_answer)

    with st.form("feedback_form"):
        st.radio("Was this answer helpful?", ["Yes", "No"], key="feedback")
        feedback_submitted = st.form_submit_button("Submit Feedback")
        if feedback_submitted:
            submit_feedback()


   
            
   
   

