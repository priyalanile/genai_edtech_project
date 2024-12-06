# Q&A Customer Chatbot for EDTech Industry:

 @Author: Priyal Nile  

## 1. Project Description/Highlights: 
- End to End Langchain Q&A System for an e-learning company codebasics.io,
- User/Customer: Would be the students who enquire few questions about the courses being offered and Chatbot would respond the answers based on Q&A.csv dataset of this ed tech company.
- When Answer is available, The correct response would be generated.
    ![](langchain_edtech_streamlit.png)
- When Answer is not available, LLM Model won't hallucinate, as it has been addressed via prompt engineering.
    ![](langchain_edtech_streamlit2.png)

## 2. Project Structure: 
- codebasics.csv: source file having Q&A pairs for type of questions generally asked by students. 
- requirements.txt: All packages that would be required for this project.
- langchain_helper.py: main lanchain code with all modules, Loader, Vector DB, LLM, Embeddings, etc.
- main.py: Streamlit Code to run application!
- .gitignore: .env file with Google Gemini API Key!
- Q&A_Chatbot_Edtech.ipynb: Proof Of Concept to ensure all individual components of projects are working fine. Eventually, these codes were added into langchain_helper.py later.
- langchain_edtech_streamlit.png: User Input scenario when answer is available.
- langchain_edtech_streamlit1.png: User Input scenario when answer is not available.
- faiss_index: directory to store serialized data of vector DB as a one time copy. However, this wasn't used in the actual code run as it was giving some Deserialization danger policy issues.

## 3. Learnings: 
- Creation of Google Gemini API Key for Google LLMs Usage
- LangChain Framework: To develop Complex Projects integration using various LangChain Modules.
- CSVLoader from Langchain: For Loading .csv file in a document format (docs with metadata).
- Hugging Face Embeddings from Langchain: To convert source data into word embeddings.
- FAISS Vector Database from Langchain: To store the word embeddings of source data for better similarity search and context search.
- Retriever Object from Langchain -> To convert user queries into vector embeddings & allow comparison of those user question embeddings with Q&A CSV word embeddings stored in vector DB and fetch relevant documents which may have the information available.
- Prompt Template from Langchain -> To avoid any hallucination (i.e. if not sure about answer, what to respond) and give proper instruction to LLM how to give output with context
- LLM (Google's Gemini Pro) from Langchain -> as an LLM to allow a proper human like response back to user following prompt

- Additional Learnings:
    - Git & GitHub Setup & Commands,
    - VSCode Setup and Python ENV Configuration,
    - Streamlit: User Interface for Quick Proof Of Concepts

## 4. Installation: 

1. Create Google Gemini API Key:
 - Google AI Studio -> Get API Keys -> Gemini API -> Google Cloud Console -> Credentials -> API Keys -> Name '<user provided input e.g. Generative Language API Key>' -> ACtion ->   Show Key.
    - Store the key within .env file separately created for just storing this key. (Make sure to add it into .gitignore to avoid it's loading into GitHub)


2. Clone this repository to your local machine using following OR:

```bash
git clone https://github.com/priyalanile/genai_edtech_project.git
```
OR Run following commands in VSCODE -> Project folder (that you created in your system) -> Terminal (Powershell) after you've Logged in to Github & created a Public/Private repository (without README.md file). 
Later, Check by refreshing browser that Github Repository is updated now.

```bash
echo "# genai_edtech_project" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/priyalanile/genai_edtech_project.git
git push -u origin main
```

3. Setup the Python Environment (env) using the following commands: 
Setting up Python Virtual Environment in VSCODE -> Project folder -> Terminal (DO preferably using CMD i.e. Command Prompt)

- To create a new environment using Conda (Anaconda installation should be in place) from Command prompt in VSCode inside your project directory.

#conda create --name myenv python=3.9
```bash
conda create -p venv python==3.9 -y 
```
- To check if the environment is created or which all envs are present:
```bash
conda info --envs
```

- To activate this newly created environment using Terminal -> CMD & not powershell: (Note: Generally you get this command ready as a part of venv creation log in above step)

```bash
#conda activate C:\Users\priya\VS_Code_Projects\5_GENAI_Edtech_VSCode_Project\venv    
conda activate ./venv 

```

- Incase, you want to deactivate the environment: 

```bash
conda deactivate 
```

- Now when within the env environment, if need to install all python libraries present in requirements.txt: 

```bash
pip install -r requirements.txt 
```

- To check which libraries are installed (within Powershell Terminal of VSCode: 
```bash
pip list | Select-String -Pattern "langchain|langchain_community|google-cloud-aiplatform|google.generativeai|scikit-learn|torch|python-dotenv|ipykernel|streamlit|tiktoken|faiss-cpu|protobuf|google.generativeai|sentence-transformers|InstructorEmbedding|langchain-google-genai|langchain-huggingface"
```
(C:\Users\priya\VS_Code_Projects\5_GENAI_Edtech_VSCode_Project\venv) PS C:\Users\priya\VS_Code_Projects\5_GENAI_Edtech_VSCode_Project>
(C:\Users\priya\VS_Code_Projects\5_GENAI_Edtech_VSCode_Project\venv) PS C:\Users\priya\VS_Code_Projects\5_GENAI_Edtech_VSCode_Project> pip list | Select-String -Pattern "langchain|python-dotenv|streamlit|tiktoken|faiss-cpu|protobuf"


## 5. Usage: 
1. Make sure the Python Environment we created is activated.
```bash
conda activate ./venv
```
2. Run the Streamlit App
```bash
streamlit run main.py
```

## 6. Possible Future Improvements In this Project: 
- Trying other models for Optimization as the current model Gemini Pro was not able to give great contextual results. e.g. It needed more dedicated questions to be answered than being open for more creative questions. 
    - Use of Open AI can address this, however, being not free, it wasn't used!

----------------------------------------------------------------
----------------------------------------------------------------

### Additional Important Links:

1. LangChain Google AI Components: https://python.langchain.com/docs/integrations/llms/google_ai/
2. Google Gemini API Key Details: https://ai.google.dev/gemini-api/docs




