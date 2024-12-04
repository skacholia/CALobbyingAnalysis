import pandas as pd
import google.generativeai as genai
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
from random import uniform
from typing import List, TypedDict, Dict, Any
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bill_summarization.log'),
        logging.StreamHandler()
    ]
)

# Define the schema for structured output
class BillAnalysis(TypedDict):
    summary: str
    topics: List[str]

def setup_gemini(api_key: str) -> genai.GenerativeModel:
    """Configure and return Gemini model"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def summarize_bill(text: str, model: genai.GenerativeModel) -> Dict[str, Any]:
    """Summarize a single bill using Gemini with structured output"""
    try:
        # Using structured output configuration
        response = model.generate_content(
            f"""Analyze this bill text and provide a detailed summary and key topics.
            Bill text: {text}""",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=BillAnalysis
            )
        )
        
        # Parse the JSON response
        try:
            result = json.loads(response.text)
            # Ensure the result has the expected structure
            if not isinstance(result, dict) or 'summary' not in result or 'topics' not in result:
                raise ValueError("Invalid response structure")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error parsing response: {str(e)}")
            return {
                "summary": f"Error parsing response: {str(e)}",
                "topics": ["Error parsing response"]
            }
            
    except Exception as e:
        logging.error(f"Error in summarize_bill: {str(e)}")
        return {
            "summary": f"Error occurred during summarization: {str(e)}",
            "topics": ["Error processing bill"]
        }

def process_batch(batch_data: List[tuple], api_key: str) -> List[tuple]:
    """Process a batch of bills"""
    model = setup_gemini(api_key)
    results = []
    
    for idx, text in batch_data:
        try:
            # Add random delay between requests
            sleep(uniform(0.5, 1.5))
            
            # Skip empty or invalid text
            if not isinstance(text, str) or not text.strip():
                results.append((idx, {
                    "summary": "No text provided",
                    "topics": ["Empty or invalid text"]
                }))
                continue
                
            summary = summarize_bill(text, model)
            results.append((idx, summary))
            
        except Exception as e:
            logging.error(f"Error processing bill at index {idx}: {str(e)}")
            results.append((idx, {
                "summary": f"Error occurred during processing: {str(e)}",
                "topics": ["Processing error"]
            }))
    
    return results

def create_output_dataframe(df: pd.DataFrame, results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Create output DataFrame with properly handled columns"""
    output_df = df.copy()
    
    # Convert results to separate series
    summaries = pd.Series({idx: res['summary'] for idx, res in results.items()})
    topics = pd.Series({idx: json.dumps(res['topics']) for idx, res in results.items()})
    
    # Add new columns
    output_df['gemini_summary'] = summaries
    output_df['gemini_topics'] = topics
    
    return output_df

def main():
    # Configuration
    BATCH_SIZE = 10  # Number of bills per batch
    MAX_WORKERS = 4  # Number of parallel processes
    SAVE_FREQUENCY = 20  # Save after processing this many bills
    INPUT_FILE = "ca_bills_data.csv"
    OUTPUT_FILE = "ca_bills_summarized.csv"
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    logging.info("Starting bill summarization process")
    
    # Read input data
    try:
        df = pd.read_csv(INPUT_FILE)
        logging.info(f"Loaded {len(df)} bills from {INPUT_FILE}")
    except Exception as e:
        logging.error(f"Error loading input file: {str(e)}")
        raise
    
    # Prepare batches
    bill_data = list(enumerate(df['full_text']))
    batches = [bill_data[i:i + BATCH_SIZE] for i in range(0, len(bill_data), BATCH_SIZE)]
    
    # Initialize results storage
    all_results = {}
    processed_count = 0
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, batch, api_key): batch_idx 
                  for batch_idx, batch in enumerate(batches)}
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            
            # Store results
            for idx, summary in batch_results:
                all_results[idx] = summary
                processed_count += 1
            
            # Periodic saving
            if processed_count % SAVE_FREQUENCY == 0:
                try:
                    output_df = create_output_dataframe(df, all_results)
                    output_df.to_csv(OUTPUT_FILE, index=False)
                    logging.info(f"Saved progress after processing {processed_count} bills")
                except Exception as e:
                    logging.error(f"Error saving progress: {str(e)}")
    
    # Final save
    try:
        output_df = create_output_dataframe(df, all_results)
        output_df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Summarization process completed")
        
        # Log summary statistics
        error_count = sum(1 for r in all_results.values() if any(
            err_text in r.get('summary', '') 
            for err_text in ['Error occurred', 'No text provided', 'Error parsing']
        ))
        logging.info(f"""
        Processing Summary:
        - Total bills processed: {len(all_results)}
        - Successful: {len(all_results) - error_count}
        - Errors: {error_count}
        - Output saved to: {OUTPUT_FILE}
        """)
        
    except Exception as e:
        logging.error(f"Error in final save: {str(e)}")
        raise

if __name__ == "__main__":
    main()