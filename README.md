# Job-Description-Search-Tool

## Quick Overview

This server acts as a **MCP tool** that searches for job descriptions on LinkedIn using a resume as input. It leverages **Ollama** for embeddings and **ChromaDB** to store and query job descriptions. The tool compares the resume against preprocessed job data to return **similar job descriptions** based on semantic similarity. It is built using fastMCP.


## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```
#### Run the Server
Run the server using the following command:
```bash
uv --directory /path/to/yourrepo run server.py
```
Replace `/path/to/yourrepo` with the actual path to your cloned repository.

### Quick setup
```json
{
  "mcpServers": {
    "job-description-search-tool": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/yourrepo",
        "run",
        "server.py"
      ],
      "env": {
        "OLLAMA_MODEL": "embedding-model",
        "LINKEDIN_EMAIL": "your_email@example.com",
        "LINKEDIN_PASSWORD": "your_secure_password",
        "LINKEDIN_BATCH_SIZE": "20",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

## Usage

1. Set environment variables:
```bash
export OLLAMA_MODEL="embedding-model"
export LINKEDIN_EMAIL="your_email@example.com"
export LINKEDIN_PASSWORD="your_password"
export LINKEDIN_BATCH_SIZE=20
export OLLAMA_URL="http://localhost:11434"
```
`LINKEDIN_BATCH_SIZE` is the number of linkedin results for each linkedin query.


2. Start the server:
```bash
uv --directory /path/to/yourrepo run server.py
```

3. Connect to the server via an MCP client using stdio