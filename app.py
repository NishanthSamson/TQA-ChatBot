import streamlit as st
from transformers import pipeline
import pandas as pd

# Load the table data


@st.cache_data
def load_table():
    url = 'https://docs.google.com/spreadsheets/d/1e9SZ2KJGsIuQEq87KziNkrp3p7RqIQRI2gHQmJ4LiN4/export?gid=1348406830&format=csv'
    table = pd.read_csv(url)
    return table.astype('str')

# Load the Hugging Face pipeline


@st.cache_resource
def load_pipeline():
    return pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")


# Initialize Streamlit App
st.title("LST Data Question Answering Chatbot")
st.write("Ask questions about the data in the table.")

# Load resources
table = load_table()
tqa = load_pipeline()

st.dataframe(table)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages


def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


# Initialize the chat display
# display_messages()

# User input
if user_input := st.chat_input("Enter your query..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # with st.chat_message("user"):
    #     st.write(user_input)

    # Process the query
    try:
        response = tqa(table=table, query=user_input)
        answer = response["cells"][0] if "cells" in response and response["cells"] else "I'm sorry, I couldn't find an answer."
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # with st.chat_message("assistant"):
    #     st.write(answer)

# Keep chat messages displayed
display_messages()
