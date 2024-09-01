import streamlit as st
from dotenv import load_dotenv
import os
from huggingface_hub import login
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
import torch


load_dotenv()

# Get the Hugging Face token from the environment variable
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Authenticate with Hugging Face using the token
if huggingface_token:
    login(token=huggingface_token)
else:
    st.error("Hugging Face token not found. Please set it in the .env file.")
# Streamlit application

# def main():
#     st.title("Llama2 Q&A Assistant")

#     # Instructions
#     st.write("""
#     This application allows you to ask questions, and the Llama2 model will provide answers based on the data it has been trained on.
#     """)

#     # User input
#     user_query = st.text_input("Enter your question:")

#     if st.button("Submit"):
#         if user_query:
#             # Load documents
#             documents = SimpleDirectoryReader("data").load_data()

#             # System prompt
#             system_prompt = """
#             You are a Q&A assistant.
#             Your goal is to answer questions as accurately as possible based on the instructions and context provided.
#             """

#             # Query wrapper prompt
#             query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

#             # Initialize the Llama2 model
#             llm = HuggingFaceLLM(
#                 context_window=4096,
#                 max_new_tokens=256,
#                 generate_kwargs={"temperature": 0.0, "do_sample": False},
#                 system_prompt=system_prompt,
#                 query_wrapper_prompt=query_wrapper_prompt,
#                 tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
#                 model_name="meta-llama/Llama-2-7b-chat-hf",
#                 device_map="auto",
#                 model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
#             )

#             # Embedding model
#             embed_model = LangchainEmbedding(
#                 HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#             )

#             # Update settings
#             from llama_index.core import Settings
#             Settings.chunk_size = 1024
#             Settings.llm = llm
#             Settings.embed_model = embed_model

#             # Create index from documents
#             index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=llm)

#             # Query the index
#             query_engine = index.as_query_engine()
#             response = query_engine.query(user_query)

#             # Display the response
#             st.write("### Response")
#             st.write(response)

#         else:
#             st.warning("Please enter a question to submit.")

# if __name__ == "__main__":
#     main()
# Streamlit application
def main():
    st.title("Llama2 Q&A Assistant")

    # Instructions
    st.write("""
    This application allows you to ask questions, and the Llama2 model will provide answers based on the data it has been trained on.
    """)

    # User input
    user_query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if user_query:
            # Load documents
            documents = SimpleDirectoryReader("data").load_data()

            # System prompt
            system_prompt = """
            You are a Q&A assistant.
            Your goal is to answer questions as accurately as possible based on the instructions and context provided.
            """

            # Query wrapper prompt
            query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

            # Initialize the Llama2 model with CPU fallback
            llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.0, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                device_map="cpu",  # Force the model to use CPU
                model_kwargs={"torch_dtype": torch.float32}  # Use full precision on CPU
            )

            # Embedding model
            embed_model = LangchainEmbedding(
                HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            )

            # Update settings
            from llama_index.core import Settings
            Settings.chunk_size = 1024
            Settings.llm = llm
            Settings.embed_model = embed_model

            # Create index from documents
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=llm)

            # Query the index
            query_engine = index.as_query_engine()
            response = query_engine.query(user_query)

            # Display the response
            st.write("### Response")
            st.write(response)

        else:
            st.warning("Please enter a question to submit.")

if __name__ == "__main__":
    main()