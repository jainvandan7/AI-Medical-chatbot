from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os 
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load and process documents
print("Loading PDF documents...")
extracted_data = load_pdf_file(data=r'c:\medical-chatbot\Data')
print(f"Loaded {len(extracted_data)} documents")

# Split text with medical-specific chunking
print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Created {len(text_chunks)} text chunks")

# Initialize embeddings
print("Initializing embeddings...")
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name
index_name = "medichatbot"

# List existing indexes
existing_indexes = pc.list_indexes().names()

# Check if the index exists before creating it
if index_name not in existing_indexes:
    print(f"Creating index {index_name}...")
    # Create index with serverless configuration
    pc.create_index(
        name=index_name,
        dimension=384,  # Matches all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    print("Waiting for index initialization...")
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(5)
    print("Index is ready!")

# Store documents in Pinecone with optimized settings
print("Storing documents in Pinecone (this may take several minutes)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace="medical_data",
    batch_size=128,
    text_key="text",
    metadata_config={
        "indexed": ["source", "page"]  # Optimize these metadata fields for filtering
    }
)

# Verify storage
index_stats = pc.describe_index(index_name)
print("\n✅ Documents successfully stored in Pinecone!")
print(f"Index Stats: {index_stats.status}")
print(f"Total vectors stored: {pc.index(index_name).describe_index_stats()['total_vector_count']}")