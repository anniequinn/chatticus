import streamlit as st

import os
import pandas as pd
import numpy as np

from scipy import spatial
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage


def compute_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)


api_key = os.getenv("OPENAI_API_KEY")  # Save to secrets

chat_model_name = "gpt-3.5-turbo"
embed_model_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model_name, api_key=api_key)
chat = ChatOpenAI(model_name=chat_model_name)

db = pd.read_parquet("data/vector_db.parquet")

st.title("Chatticus\nThe Atticus Journal virtual assistant")

with st.form("my_form"):

    # Allow user to set n_queries via a slider, min 1, max 25
    n_chunks = st.slider(
        "Select the number of chunks to retrive",
        min_value=1,
        max_value=25,
        value=5,
    )

    # Build the query context
    context_default = """You are a virtual assistant specifically designed to provide information about WPP's Atticus Journal, Volume 28. You have access to a knowledge base of the specific articles within this edition of the journal. 
Your purpose is to assist users with inquiries directly related to this journal, including discussions on marketing strategies, creative insights, and case studies presented within this volume.
When users ask questions, your responses should be accurate and relevant to the information contained in Atticus Journal, Volume 28. If a user asks a question that falls outside the scope of this journal or relates to general topics not covered within it, you should respond with: 'Sorry, I can't help with that. I can only answer questions relating to the Atticus Journal, Volume 28. Please feel free to ask about the content, themes, or specific articles within this edition.'
Remember, your goal is to be helpful and informative about the Atticus Journal, Volume 28, while clearly communicating the boundaries of your expertise.
    """

    context = st.text_area(
        "Context (update optional)",
        context_default,
        max_chars=1500,
        height=250,
    )
    messages = [SystemMessage(content=context)]

    # Allow user to input a query - their chosen question
    user_query_default = (
        "What does the journal say about multicrises in marketing?"
    )
    user_query = st.text_area(
        "Your query goes here", user_query_default, max_chars=250
    )

    submitted = st.form_submit_button("Search")
    # Show a spinner while the model is working
    with st.spinner("Searching for the most relevant information..."):

        if submitted:
            # Embed the user query
            embedded_query = embeddings.embed_query(user_query)

            # Use cosine similarity to get the n most similar texts
            similarities_array = np.array(
                [
                    compute_similarity(embedded_query, embedding)
                    for embedding in db["embedding"].values
                ]
            )
            top_indices = np.argsort(similarities_array)[::-1][:n_chunks]
            matches = db.iloc[top_indices]

            # Append the user query to messages
            appended_user_query = user_query
            appended_user_query += "Based on the information retrieved, consider the following text to guide your response:\n\n"
            appended_user_query += "\n\n".join(matches["text"].values)
            messages.append(HumanMessage(content=appended_user_query))

            # Get the response from the chat model
            response = chat(messages)
            st.markdown(response.content)
