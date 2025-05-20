from src.util import *


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# AZURE_OPEN_API_KEY, AZURE_OPEN_API_ENDPOINT, AZURE_OPEN_API_VERSION, AZURE_EMBEDDING_API_ENDPOINT, AZURE_EMBEDDING_VERSION = fetch_azure_openai_credentials()

file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(f"Number of documents loaded: {len(docs)}")

text_splitter =RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
len(all_splits)

# print(all_splits[0].page_content)


# embeddings = AzureOpenAIEmbeddings(
#     api_key=AZURE_OPEN_API_KEY,  
#     api_version=AZURE_EMBEDDING_VERSION,
#     azure_endpoint=AZURE_EMBEDDING_API_ENDPOINT,
#     deployment="text-embedding-ada-002",
#     model="text-embedding-ada-002",
#     )

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf_sent_transformer_embedding = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_1 = hf_sent_transformer_embedding.embed_query(all_splits[0].page_content)
vector_2 = hf_sent_transformer_embedding.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")


#4. Vector Store
vector_store = Chroma(
    collection_name = "nike_collection",
    embedding_function=hf_sent_transformer_embedding,
    persist_directory="./chroma_lengchain_db"
)

# add documents to the vector store
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print(results[0])


results_with_score = vector_store.similarity_search_with_score(
    "How many distribution centers does Nike have in the US?"
)

doc, score = results_with_score[0]
print(score)

embedded_input = hf_sent_transformer_embedding.embed_query(
    "How many distribution centers does Nike have in the US?"
)
results_with_embedding = vector_store.similarity_search_by_vector(
    embedded_input
)

print(results_with_embedding[0])



