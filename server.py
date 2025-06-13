from fastmcp import FastMCP, Context
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os
from typing import List, Dict, Any, Annotated
from pydantic import Field
from linkedin_wrapper import search_jobs as linkedin_search_jobs
import onnxruntime
from transformers import AutoTokenizer
import numpy as np

class ONNXEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path: str, tokenizer_name: str = "onnx-community/Qwen3-Embedding-0.6B-ONNX"):
        # Initialize ONNX runtime session
        self.session = onnxruntime.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.padding_side = 'left'
        
    def last_token_pool(self, last_hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        float_attention_mask = attention_mask.astype(np.float32)
        batch_size, seq_len, hidden_size = last_hidden_states.shape
        
        # Check if all sequences end at the same position
        if np.all(float_attention_mask[:, -1] == 1):
            return last_hidden_states[:, -1, :]
        else:
            sequence_lengths = np.sum(float_attention_mask, axis=1) - 1
            pooled_embeddings = []
            for i in range(batch_size):
                seq_len_i = int(sequence_lengths[i])
                pooled_embeddings.append(last_hidden_states[i, seq_len_i, :])
            return np.stack(pooled_embeddings)
    
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
            
        # Tokenize with all required fields
        encoded_input = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=1024
        )
        
        # Prepare ONNX inputs
        ort_inputs = {
            "input_ids": encoded_input["input_ids"],
            "attention_mask": encoded_input["attention_mask"],
            "position_ids": np.arange(encoded_input["input_ids"].shape[1]).reshape(1, -1)
        }
        
        # Run inference
        ort_outputs = self.session.run(None, ort_inputs)
        output_names = [output.name for output in self.session.get_outputs()]
        if "last_hidden_state" in output_names:
            last_hidden_state = ort_outputs[output_names.index("last_hidden_state")]
        else: 
            raise Exception(f"Last hidden state not found")
        
        # Pool embeddings
        pooled = self.last_token_pool(last_hidden_state, encoded_input["attention_mask"])
        
        # Normalize embeddings
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        
        return pooled.tolist()

env_vars = {
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
    "LINKEDIN_EMAIL": os.getenv("LINKEDIN_EMAIL"),
    "LINKEDIN_PASSWORD": os.getenv("LINKEDIN_PASSWORD"),
    "LINKEDIN_BATCH_SIZE": int(os.getenv("LINKEDIN_BATCH_SIZE", 10)), # Increase limit for better results
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
}

mcp = FastMCP("Job Description Search Tool")

def check_env():
    for var_name, var_value in env_vars.items():
        if var_value is None:
            raise ValueError(f"Environment variable {var_name} is not set.")

@mcp.tool
async def search_jobs(
    ctx: Context,
    resume: Annotated[str, Field(description="The resume text to match against.")] = "I'm a sofware engineer, c++, rust",
    queries: Annotated[List[str], Field(description="List of search queries for the input resume.")] = ["software", "devops"],
    n_results: Annotated[int, Field(description="Number of similar job descriptions to return.", default=5)] = 5,
):
    """
    Search jobs in Linkedin using the porvided queries and retrive the relevant ones with respect to the resume.
    Note that the total queries number is the length of the quesries list time 10, the sweet spot is providing 3 to 5 queries.

    Args:
        resume: Text of the resume to match against.
        queries: List of different search queries to inject in Linkedin for the input resume.
        n_results: Number of similar job descriptions to return.

    Returns:
        List of job descriptions similar to the resume.
    """
    # Initialize ChromaDB client and collection
    #ollama_ef = OllamaEmbeddingFunction(
    #    url=env_vars['OLLAMA_URL'],
    #    model_name=env_vars['OLLAMA_MODEL'],
    #)
    ollama_ef = ONNXEmbeddingFunction(model_path="model.onnx")
    chroma_client = chromadb.EphemeralClient()
    collection_name = "job_embeddings"
    collection = chroma_client.create_collection(name=collection_name, embedding_function=ollama_ef)

    # Process each query
    for i, query in enumerate(queries):
        await ctx.info(f"Processing query {query} ({i+1}/{len(queries)})")
        jobs = linkedin_search_jobs(
            keywords=query,
            limit=env_vars['LINKEDIN_BATCH_SIZE'],  
        )
        await ctx.info(jobs[0])
        collection.add(
            documents=[job["description"] for job in jobs],
            metadatas=[{
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", "")
            } for job in jobs],
            ids=[f"{job.get('title', '').replace(' ','_')}_{job.get('company','').replace(' ','_')}" for job in jobs]
        )

    # Perform similarity search
    await ctx.info(f"Querying best results ...")
    results = collection.query(
        query_texts=[resume],
        n_results=int(n_results)
    )

    chroma_client.delete_collection(name=collection_name)
    return [
        {
            "title": m['title'],
            "location": m['location'],
            "company": m['company'],
            "description": d
        }
        for m, d in zip(results["metadatas"][0], results['documents'][0])
    ] if len(results['metadatas']) > 0 else ["No matching results found"]

if __name__ == "__main__":
    check_env()
    mcp.run()
