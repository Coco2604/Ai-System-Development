import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset from the CSV file
df = pd.read_csv('movies.csv')

# Create embeddings for the movie plots
# Initialize the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each plot in the DataFrame.
# The model.encode() method takes a list of strings and returns a list of embeddings.
print("Creating embeddings for movie plots...")
plot_embeddings = model.encode(df['plot'].tolist(), convert_to_tensor=False)
print("Embeddings created successfully.")

def search_movies(query, top_n=5):
    """
    Searches for movies based on a query string.

    Args:
        query (str): The search query.
        top_n (int): The number of top results to return.

    Returns:
        pandas.DataFrame: A DataFrame containing the top_n most similar movies,
                          including their title, plot, and similarity score.
    """

    # 1. Create an embedding for the user's query
    query_embedding = model.encode([query], convert_to_tensor=False)

    # 2. Calculate cosine similarity between the query embedding and all plot embeddings
    # cosine_similarity expects 2D arrays, so we reshape the embeddings.
    similarities = cosine_similarity(query_embedding, plot_embeddings)[0]

    # 3. Get the indices of the top_n most similar movies
    # np.argsort sorts the similarities in ascending order, so we take the last 'top_n' elements.
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # 4. Create a new DataFrame with the results
    # We select the rows from the original DataFrame using the top indices.
    result_df = df.iloc[top_indices].copy()

    # Add the similarity scores to the result DataFrame
    result_df['similarity'] = [similarities[i] for i in top_indices]

    return result_df