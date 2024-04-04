"""
Simple script to embed the text data in the data directory and save it as a 
local equivalent of a vector database. Only needs to be run once to create the
vector database. The vector database is saved as a parquet file in `data`.
"""

import os
import textract
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
embed_model_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model_name, api_key=api_key)

n_chunks = 2  # Number of chunks to overlap
data = []

# Get the files in the data directory
data_dir = os.path.expanduser("~/git/personal/chatticus/data")
files = os.listdir(data_dir)

for file_name in files:
    # Get the text from the file
    file_path = os.path.join(data_dir, file_name)
    file_name_clean = file_name.split("|")[0].strip()
    extracted_text = textract.process(file_path)
    decoded_text = extracted_text.decode("utf-8")

    # Split text into paragraphs and clean up
    split_text = decoded_text.split("\n\n")  # Split on paragraphs
    split_text = split_text[9:-12]  # Remove header and footer
    split_text = [text.replace("\n", " ") for text in split_text]  # Remove line breaks

    # Overlap chunks of text - n chunks before and after
    for i in range(n_chunks, len(split_text) - n_chunks):
        text = " ".join(split_text[i - n_chunks:i + n_chunks + 1])
        metadata = {"file_name": file_name_clean, "chunk_number": i}
        data.append({"text": text, "metadata": metadata})

data = pd.DataFrame(data)

# Embed the text
data["embedding"] = embeddings.embed_documents(data["text"].tolist())

# Save as a local equivalent of a vector database
data.to_parquet("data/vector_db.parquet")