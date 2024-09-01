# import streamlit as st
# import os
# import snowflake.connector
# import pandas as pd
# from dotenv import load_dotenv
# from llama_index.llms.openai import OpenAI
# from llama_index.core import VectorStoreIndex, Document

# # Load environment variables
# load_dotenv()

# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# # Set up Snowflake connection parameters
# SNOWFLAKE_ACCOUNT= os.getenv("SNOWFLAKE_ACCOUNT")
# SNOWFLAKE_USER= os.getenv("SNOWFLAKE_USER")
# SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
# SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
# SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
# SNOWFLAKE_WAREHOUSE= os.getenv("SNOWFLAKE_WAREHOUSE")
# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

# # Set up Snowflake connection parameters
# # os.environ['SNOWFLAKE_ACCOUNT']= os.getenv("SNOWFLAKE_ACCOUNT")
# # os.environ['SNOWFLAKE_USER']= os.getenv("SNOWFLAKE_USER")
# # os.environ['SNOWFLAKE_PASSWORD']= os.getenv("SNOWFLAKE_PASSWORD")
# # os.environ['SNOWFLAKE_DATABASE']= os.getenv("SNOWFLAKE_DATABASE")
# # os.environ['SNOWFLAKE_SCHEMA']= os.getenv("SNOWFLAKE_SCHEMA")
# # os.environ['SNOWFLAKE_WAREHOUSE']= os.getenv("SNOWFLAKE_WAREHOUSE")

# # SNOWFLAKE_USER="anush",
# # SNOWFLAKE_PASSWORD="Dolo650$$",
# # SNOWFLAKE_ACCOUNT="TEB16561",
# # SNOWFLAKE_WAREHOUSE="SNOWFLAKE_WAREHOUSE",
# # SNOWFLAKE_DATABASE="SNOWFLAKE_DATABASE",
# # #role="ACCOUNTADMIN",
# # SNOWFLAKE_SCHEMA="",


# # Set page configuration and initialize OpenAI API key
# st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

# st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

# if "messages" not in st.session_state.keys():  # Initialize the chat messages history
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
#     ]

# @st.cache_resource(show_spinner=False)
# def load_data():
#     with st.spinner(text="Loading and indexing data from Snowflake – hang tight! This should take 1-2 minutes."):
#         # Connect to Snowflake
#         conn = snowflake.connector.connect(
#             user=SNOWFLAKE_USER,
#             password=SNOWFLAKE_PASSWORD,
#             account=SNOWFLAKE_ACCOUNT,
#             warehouse=SNOWFLAKE_WAREHOUSE,
#             database=SNOWFLAKE_DATABASE,
#             schema=SNOWFLAKE_SCHEMA
#         )

#         # Execute a query to fetch data
#         query = "SELECT * FROM parsed_doc_chunks"  # Replace with your actual SQL query
#         data = pd.read_sql(query, conn)

#         # Close the connection
#         conn.close()

#         # Convert DataFrame rows to Document objects
#         docs = [Document(content=row.to_json()) for _, row in data.iterrows()]

#         # Initialize the OpenAI LLM directly
#         llm = OpenAI(model="gpt-4o-mini", temperature=0.5, 
#                      system_prompt="You are an expert on code documentation and your job is to answer technical questions. Assume that all questions are related to the code shared in context. Keep your answers technical and based on facts – do not hallucinate features.")
        
#         # Directly pass the LLM instance when creating the index
#         index = VectorStoreIndex.from_documents(docs, llm=llm)
#         return index

# index = load_data()

# if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
#     st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

# if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

# for message in st.session_state.messages:  # Display the prior chat messages
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.chat_engine.chat(prompt)
#             st.write(response.response)
#             message = {"role": "assistant", "content": response.response}
#             st.session_state.messages.append(message)  # Add response to message history

        

# import streamlit as st
# import os
# import snowflake.connector
# import pandas as pd
# from dotenv import load_dotenv
# from llama_index.llms.openai import OpenAI
# from llama_index.core import VectorStoreIndex, Document

# # Load environment variables
# load_dotenv()

# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# # Set up Snowflake connection parameters
# SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
# SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
# SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
# SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
# SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
# SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

# # Set page configuration and initialize OpenAI API key
# st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

# st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

# if "messages" not in st.session_state.keys():  # Initialize the chat messages history
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
#     ]

# @st.cache_resource(show_spinner=False)
# def load_data():
#     with st.spinner(text="Loading and indexing data from Snowflake – hang tight! This should take 1-2 minutes."):
#         # Connect to Snowflake
#         conn = snowflake.connector.connect(
#             user=SNOWFLAKE_USER,
#             password=SNOWFLAKE_PASSWORD,
#             account=SNOWFLAKE_ACCOUNT,
#             warehouse=SNOWFLAKE_WAREHOUSE,
#             database=SNOWFLAKE_DATABASE,
#             schema=SNOWFLAKE_SCHEMA
#         )

#         # Execute a query to fetch data
#         query = "SELECT * FROM parsed_doc_chunks"  # Replace with your actual SQL query
#         data = pd.read_sql(query, conn)

#         # Close the connection
#         conn.close()

#         # Check the DataFrame content and columns
#         print(data.head())  # Check the first few rows of your DataFrame
#         print(data.columns)  # Print column names to help find the correct one

#         # Replace 'your_text_column_name' with the actual column name containing the text content
#         text_column_name = 'CHUNK'  # Update this line with the correct column name

#         if text_column_name not in data.columns:
#             raise KeyError(f"Column '{text_column_name}' not found in DataFrame. Available columns: {data.columns.tolist()}")

#         # Convert DataFrame rows to Document objects and filter out empty content
#         docs = [
#             Document(content=row[text_column_name]) 
#             for _, row in data.iterrows() 
#             if pd.notnull(row[text_column_name]) and row[text_column_name].strip() != ""
#         ]

#         # Debugging: Check the number of documents created
#         print(f"Number of valid documents created: {len(docs)}")
#         for doc in docs:
#             print(doc.content)  # Print the content of each document (optional)

#         if not docs:
#             raise ValueError("No valid documents were found. Please ensure your data has content.")

#         # Initialize the OpenAI LLM directly
#         llm = OpenAI(model="gpt-4o-mini", temperature=0.5, 
#                      system_prompt="You are an expert on code documentation and your job is to answer technical questions. Assume that all questions are related to the code shared in context. Keep your answers technical and based on facts – do not hallucinate features.")
        
#         # Directly pass the LLM instance when creating the index
#         index = VectorStoreIndex.from_documents(docs, llm=llm)
#         return index

# index = load_data()


# if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
#     st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

# if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

# for message in st.session_state.messages:  # Display the prior chat messages
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.chat_engine.chat(prompt)
#             st.write(response.response)
#             message = {"role": "assistant", "content": response.response}
#             st.session_state.messages.append(message)  # Add response to message history


import streamlit as st
import os
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Document

# Load environment variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Set up Snowflake connection parameters
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

# Set page configuration and initialize OpenAI API key
st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

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

        # Execute a query to fetch data
        query = "SELECT * FROM parsed_doc_chunks"  # Replace with your actual SQL query
        data = pd.read_sql(query, conn)

        # Close the connection
        conn.close()

        # Check the DataFrame content and columns
        # print(data.head())  # Check the first few rows of your DataFrame
        # print(data.columns)  # Print column names to help find the correct one

        # Replace 'your_text_column_name' with the actual column name containing the text content
        text_column_name ='CHUNK'  # Update this line with the correct column name

        if text_column_name not in data.columns:
            raise KeyError(f"Column '{text_column_name}' not found in DataFrame. Available columns: {data.columns.tolist()}")

        # Convert DataFrame rows to Document objects and filter out empty content
        docs = [
            Document(text=row[text_column_name])  # Updated to use `text` instead of `content`
            for _, row in data.iterrows()
            if pd.notnull(row[text_column_name]) and row[text_column_name].strip() != ""
        ]

        # Debugging: Check the number of documents created
        print(f"Number of valid documents created: {len(docs)}")
        # for doc in docs:
        #     print(doc.text)  # Updated to use `text` to access the content (assuming this is correct)

        # if not docs:
        #     raise ValueError("No valid documents were found. Please ensure your data has content.")

        # Initialize the OpenAI LLM directly
        llm = OpenAI(model="gpt-4o-mini", temperature=0.5, 
                     system_prompt="You are an expert on data documentation and your job is to answer technical questions. Assume that all questions are related to the data shared in context. Keep your answers technical and based on facts – do not hallucinate features.")
        
        # Directly pass the LLM instance when creating the index
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

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history


