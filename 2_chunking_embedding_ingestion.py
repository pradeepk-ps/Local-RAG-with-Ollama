#################################################################################################################################################################
###############################   1. IMPORTING MODULES AND INITIALIZING VARIABLES   ###########################################################################
#################################################################################################################################################################

from dotenv import load_dotenv
import os
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil

load_dotenv()

#################################################################################################################################################################
###############################   2. INITIALIZE EMBEDDINGS MODEL   #############################################################################################
#################################################################################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

#################################################################################################################################################################
###############################   3. DELETE OLD CHROMA DB IF EXISTS   ##########################################################################################
#################################################################################################################################################################

if os.path.exists(os.getenv("DATABASE_LOCATION")):
    shutil.rmtree(os.getenv("DATABASE_LOCATION"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

#################################################################################################################################################################
###############################   4. INITIALIZE TEXT SPLITTER   ###############################################################################################
#################################################################################################################################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

#################################################################################################################################################################
###############################   5. PROCESS JSON FILE   ######################################################################################################
#################################################################################################################################################################

def process_json_lines(file_path):
    extracted = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                extracted.append(obj)
            except Exception as e:
                print("Skipping invalid JSON line:", e)

    return extracted


file_path = os.path.join(
    os.getenv("DATASET_STORAGE_FOLDER"),
    "data.txt"
)

file_content = process_json_lines(file_path)

#################################################################################################################################################################
###############################   6. CHUNKING, EMBEDDING, AND INGESTION   ####################################################################################
#################################################################################################################################################################

for line in file_content:

    # Get raw text safely
    raw_text = line.get("raw_text", "")

    # Skip empty records
    if not raw_text or not raw_text.strip():
        print("Skipping empty record")
        continue

    # Print current URL
    print(line.get("url", "No URL Found"))

    try:

        # Create chunks
        texts = text_splitter.create_documents(
            [raw_text],
            metadatas=[{
                "source": line.get("url", ""),
                "title": line.get("title", "")
            }]
        )

        # Skip if no chunks created
        if not texts:
            print("No chunks created")
            continue

        # Generate UUIDs
        uuids = [str(uuid4()) for _ in range(len(texts))]

        # Add to ChromaDB
        vector_store.add_documents(
            documents=texts,
            ids=uuids
        )

        print(f"Successfully added {len(texts)} chunks")

    except Exception as e:
        print("Error processing record:", e)
        continue

#################################################################################################################################################################
###############################   7. FINISHED   ################################################################################################################
#################################################################################################################################################################

print("\nAll data successfully embedded into ChromaDB 🚀")