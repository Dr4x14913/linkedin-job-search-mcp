from fastmcp import FastMCP, Context
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os
from typing import List, Dict, Any, Annotated
from pydantic import Field
from linkedin_wrapper import search_jobs as linkedin_search_jobs
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

class Qwen3_0p6B_ONNXEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Initialize ONNX runtime session
        hf_repo = "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8"
        model_path = hf_hub_download(hf_repo, "dynamic_uint8.onnx")
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        self.tokenizer.padding_side = 'left'
        
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
        inputs = {}
        for inp in self.session.get_inputs():
            if inp.name in ort_inputs:
                inputs[inp.name] = ort_inputs[inp.name].astype(np.int64)

        # Run inference
        ort_outputs = self.session.run(None, inputs)
        pooled = ort_outputs[0]
        
        # Normalize embeddings
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        print(pooled) 
        
        return pooled.tolist()

env_vars = {
    "LINKEDIN_EMAIL": os.getenv("LINKEDIN_EMAIL"),
    "LINKEDIN_PASSWORD": os.getenv("LINKEDIN_PASSWORD"),
    "LINKEDIN_BATCH_SIZE": int(os.getenv("LINKEDIN_BATCH_SIZE", 10)), # Increase limit for better results
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
    ef = Qwen3_0p6B_ONNXEmbeddingFunction()
    chroma_client = chromadb.EphemeralClient()
    collection_name = "job_embeddings"
    collection = chroma_client.create_collection(name=collection_name, embedding_function=ef,  metadata={"hnsw:space": "cosine"})

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
