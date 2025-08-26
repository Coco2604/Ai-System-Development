# Semantic Search on Movie Plots Assignment

This repository contains a Python-based semantic search engine for movie plots. It uses the `sentence-transformers` library to generate vector embeddings for movie plots and calculates cosine similarity to find movies relevant to a user's query.

## Setup

Follow these steps to set up the project on your local machine.

1. Clone the repository:
    ```bash
    git clone https://github.com/avanig1834/AI-Systems-Development.git
    cd Assignment-1/
    ```

2. Create and activate a virtual environment:
    * On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    * On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the script:
    To see the default example search, run the main script:
    ```bash
    python movie_search.py
    ```
 
# Testing

To run the automated unit tests and verify that the search functionality is working correctly, use the following command from the root directory of the project:

```bash
python -m unittest discover -v
```

You should see output indicating that all tests passed successfully.

# Usage

The core of the project is the `search_movies()` function. You can import and use it in your own scripts.

Example:

Create a Python file (e.g., `main.py`) and add the following code:

```python
# Import the function from the movie_search module
from movie_search import search_movies

# Define your query
my_query = 'a spy thriller in Paris'

# Get the top 3 results
results = search_movies(query=my_query, top_n=3)

# Print the results
print(f"Search results for: '{my_query}'")
print(results)
```
