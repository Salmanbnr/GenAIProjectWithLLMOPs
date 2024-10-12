from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import boto3
import os

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Optimized prompt template
prompt_template = """
Human: Use the following context to provide an accurate and concise answer to the question.
If the question requires detail, provide a well-explained response of appropriate length.
If you don't know the answer, say that you don't know—don't make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def data_ingestion():
    loader = PyPDFDirectoryLoader("./Data")
    document = loader.load()
    # Optimized text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(document)
    return docs

def get_vector_store(docs, rebuild=False):
    if os.path.exists("faiss_index") and not rebuild:
        return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vector_store_faiss.save_local("faiss_index")
        return vector_store_faiss

def get_model():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock)
    return llm

def get_response_llm(llm, vector_store_faiss, query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store_faiss.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        return answer["result"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    docs = data_ingestion()
    vectorstore_faiss = get_vector_store(docs)
    query = "who created this chatbot"
    llm = get_model()
    print(get_response_llm(llm, vectorstore_faiss, query))
