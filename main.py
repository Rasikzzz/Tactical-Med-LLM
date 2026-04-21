# Install zstd and then run the Ollama installer
!apt-get update && apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

# 1. INSTALLATION & OLLAMA SETUP
!pip install langchain langchain-community langchain-ollama chromadb pypdf sentence-transformers
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time, os
subprocess.Popen(['ollama', 'serve'])
time.sleep(10)
!ollama pull llama3

# DOCUMENT PROCESSING
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# Load the ATP 4-02.11 PDF
loader = PyPDFLoader("manual.pdf")
data = loader.load()

# Split into tactical chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# 3. VECTOR STORAGE (The Brain)
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cuda'} # Uses Colab GPU
)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Your Custom Military Prompt Template
template = """Use the following pieces of military documentation to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and follow standard military terminology.

{context}

Question: {question}
Helpful Answer:"""

# We use ChatPromptTemplate or PromptTemplate from langchain_core
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 2. Initialize the Model
llm = OllamaLLM(model="llama3")

# 3. Define how to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Construct the Chain (This replaces RetrievalQA)
# This "pipes" the data from the retriever, through the prompt, to the LLM
qa_chain = (
    {
        "context": vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs,
        "question": RunnablePassthrough()
    }
    | QA_CHAIN_PROMPT
    | llm
    | StrOutputParser()
)

print("✅ Military QA Chain is ready!")

query = "What are the precise steps for performing a needle chest decompression as outlined in the TCCC section?"
response = qa_chain.invoke(query)
print(response)
