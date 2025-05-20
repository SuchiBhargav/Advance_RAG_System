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


import streamlit as st
import os
import datetime
import json


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]= "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"

#loader = PyPDFLoader("sample.pdf")
loader = TextLoader("sample.txt")
docs = loader.load()

#For large language models, it's not strictly required, but it can reduce noise and help with consistency, especially in RAG indexing.
for doc in docs:
    doc.page_content = doc.page_content.lower()




text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # increase size
    chunk_overlap=200,  # more overlap between chunks
    separators=["\n\n", "\n", ".", " ", ""],  # respects logical breaks)
)

documents=text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="llama3")

# This embeds the documents and builds the vector store
db = FAISS.from_documents(documents, embeddings)

# # Set up retriever Lower k to reduce noise.
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})  # k increase you're getting more context chunks

# Define the LLM
#Low temperature ensures the model sticks closely to retrieved documents rather than inventing.
llm = OllamaLLM(model="llama3",temperature=0.1,stream=True)
# steam - This will enable streaming output, meaning tokens will be sent as they are generated (like a live typing effect), which improves perceived performance.

## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a technical assistant. You must answer **only** using the context provided.
Do not guess or add details not explicitly stated.
                                          
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


# ## streamlit framework

st.title('Welcome to the MgenAi chatbot \U0001F916 ')
input_text=st.text_input("Are you stuck with the cleanup or stage1 ?? Enter your question:")

if input_text:
    context_text = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(input_text)])
    response = qa_chain.invoke({
        "context": context_text,
        "input": input_text
    })
    st.write("### Answer:")   #This is a Markdown string, and the ### indicates a level-3 heading in Markdown.
    st.write(response['answer'])

    feedback = st.radio("Was this answer helpful?", ["Yes", "No"])
    submit_button = st.button("Submit Feedback")

    if submit_button:
        if feedback == "Yes":
            st.write("Thank you for the feedback! We're glad the answer was correct and useful.")
        else:
            st.write("Thank you for the feedback! We'll work on improving the answers.")
        feedback_data = {
        "timestamp": str(datetime.datetime.now()),
        "question": input_text,
        "answer": response['answer'],
        "feedback": feedback
    }

        # Store feedback (JSON or CSV)
        with open("feedback_log.json", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
            
   
   

