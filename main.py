#from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader

#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv #Getting KEY setup for LLM:
import os
load_dotenv() # Load the .env file


llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"), temperature=0.0) #Temp 0 as we don't want model to be creative but provide answer based on Q&A csv only.



sentence_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectordb_file_path = "faiss_index"

loader=CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
data=loader.load()

# def create_vector_db():
#     """
#     As we don't want everytime to create the vector DB being time consuming process.
    
#     """
#     loader=CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
#     data=loader.load()
#     vectordb = FAISS.from_documents(documents=data, embedding=sentence_embeddings)
#     vectordb.save_local(vectordb_file_path)

def get_qa_chain():

    #vectordb=FAISS.load_local(vectordb_file_path,sentence_embeddings) as it was giving deserialization warnings!
    vectordb = FAISS.from_documents(documents=data, embedding=sentence_embeddings)

    retriever = vectordb.as_retriever()
    prompt_template = """Given the following context and a question, generate an answer based on this context only. In the answer try to provide as much text as possible from "response" section in the source document without making things up.
    if the answer is not found in the context, kindly state "I am unable to provide this information. Kindly get in touch with abc_helpdesk@gmail.com."
    Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"]
                            )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":PROMPT})
    
    return chain

if __name__ == "__main__":
    #create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you provide virtual internships or job assistance?"))