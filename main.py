"""
This would contain the streamlit code+ Langchain code from helper.py
"""
import streamlit as st
from langchain_helper import get_qa_chain

st.title("Q&A Chatbot for EdTech Industry 👨‍💻🎓")


# btn = st.button("Create Knowledgebase")

# if btn:
#     pass

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Asnwer: ")
    st.write(response["result"])