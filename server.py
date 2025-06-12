from fastmcp import FastMCP, Context
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os
from typing import List, Dict, Any, Annotated
from pydantic import Field
from linkedin_wrapper import search_jobs as linkedin_search_jobs

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
    resume: Annotated[str, Field(description="The resume text to match against.")],
    queries: Annotated[List[str], Field(description="List of search queries for the input resume.")],
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
    # Initialize Ollama embedding function
    ollama_ef = OllamaEmbeddingFunction(
        url=env_vars['OLLAMA_URL'],
        model_name=env_vars['OLLAMA_MODEL'],
    )

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.Client()
    collection_name = "job_embeddings"
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=ollama_ef)

    # Process each query to add job descriptions
    for i, query in enumerate(queries):
        await ctx.info(f"Processing query {query} ({i+1}/{len(queries)})")
        jobs = linkedin_search_jobs(
            keywords=query,
            limit=env_vars['LINKEDIN_BATCH_SIZE'],  
        )
        
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

    # delete collection
    chroma_client.delete_collection(name=collection_name)

    await ctx.info(f"{results}")
    # Return the matched job descriptions
    return [
        {
            "title": m['title'],
            "location": m['location'],
            "company": m['company'],
            "description": d
        }
        for m, d in zip(results["metadatas"][0], results['documents'][0])
    ]

if __name__ == "__main__":
    check_env()
    mcp.run()
