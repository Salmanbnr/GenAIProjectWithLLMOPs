import warnings
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from QAsystem.retreivalandgeneration import get_model, get_response_llm

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock)


def main():

    st.set_page_config("Course Chatbot")
    st.header("CS Courses ChatBot")
    user_question = st.text_input("Ask about courses")

    if st.button("Search"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

            llm = get_model()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == '__main__':
    main()
