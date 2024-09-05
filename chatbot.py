import streamlit as st
import os
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Document


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")


st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Hi there👋 Welcome!!💬🦙")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions about the data present in the documents💬"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing data from Snowflake – hang tight! This should take few seconds."):
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        query = "SELECT * FROM parsed_doc_chunks"  
        data = pd.read_sql(query, conn)
        conn.close()


        # print(data.head())  
        # print(data.columns) 

        
        text_column_name ='CHUNK'  

        if text_column_name not in data.columns:
            raise KeyError(f"Column '{text_column_name}' not found in DataFrame. Available columns: {data.columns.tolist()}")

        # Convert DataFrame rows to Document objects and filter out empty content
        docs = [
            Document(text=row[text_column_name])  
            for _, row in data.iterrows()
            if pd.notnull(row[text_column_name]) and row[text_column_name].strip() != ""
        ]

     
        print(f"Number of valid documents created: {len(docs)}")
        # for doc in docs:
        #     print(doc.text)  # Updated to use `text` to access the content (assuming this is correct)

        # if not docs:
        #     raise ValueError("No valid documents were found. Please ensure your data has content.")

    
        llm = OpenAI(model="gpt-4o-mini", temperature=0.5, 
                     system_prompt="You are an expert on data documentation and your job is to answer technical questions. Assume that all questions are related to the data shared in context. Keep your answers technical and based on facts – do not hallucinate features.")
    
        index = VectorStoreIndex.from_documents(docs, llm=llm)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  


