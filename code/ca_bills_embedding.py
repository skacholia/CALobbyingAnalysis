import pandas as pd
import numpy as np
from openai import OpenAI
import os
import time
from tqdm import tqdm
import asyncio
import aiohttp
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to get embeddings in batches
async def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    """
    Get embeddings for a list of texts in batches
    Returns: List of embeddings in the same order as input texts
    """
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            # Extract embeddings in correct order
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            # Rate limiting - sleep for a short time between batches
            
        except Exception as e:
            print(f"Error in batch starting at index {i}: {str(e)}")
            # On error, append None for each text in the failed batch
            embeddings.extend([None] * len(batch))
    
    return embeddings

async def main():
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv('ca_bills_summarized.csv')
    
    # Get embeddings for summaries
    print("Getting embeddings...")
    summaries = df['gemini_summary'].fillna('').tolist()
    embeddings = await get_embeddings_batch(summaries)
    
    # Add embeddings to dataframe
    print("Adding embeddings to dataframe...")
    df['summary_embedding'] = embeddings
    
    # Save to new CSV
    print("Saving to CSV...")
    output_filename = 'ca_bills_summarized_with_embeddings.csv'
    df.to_csv(output_filename, index=False)
    print(f"Saved to {output_filename}")
    
    # Print some stats
    print("\nStats:")
    print(f"Total rows processed: {len(df)}")
    print(f"Rows with embeddings: {df['summary_embedding'].notna().sum()}")
    print(f"Rows without embeddings: {df['summary_embedding'].isna().sum()}")

if __name__ == "__main__":
    asyncio.run(main())