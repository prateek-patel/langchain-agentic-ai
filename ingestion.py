import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("/Users/prateekpatel/Workspace/langchain-udemy/langchain-agentic-ai/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Embedding means converting text to vector 
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")