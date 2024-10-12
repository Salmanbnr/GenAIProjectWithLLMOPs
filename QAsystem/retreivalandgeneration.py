from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms.bedrock import Bedrock
import boto3
from langchain.prompts import PromptTemplate
from ingestion import data_ingestion
from ingestion import get_vector_store
from langchain_community.embeddings import BedrockEmbeddings

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock)

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
200 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_model():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock)

    return llm


def get_response_llm(llm, vector_store_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer["result"]


if __name__ == '__main__':
    docs=data_ingestion()
    vectorstore_faiss=get_vector_store(docs)
    faiss_index = FAISS.load_local(
        "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    query = "what is the courses in Semester 4"
    llm = get_model()
    print(get_response_llm(llm, faiss_index, query))
