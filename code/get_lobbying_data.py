import pandas as pd
import requests
import os
import google.generativeai as genai
import logging
from time import sleep
from random import uniform
import json
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

def download_pdf(url: str, output_path: str) -> str:
    """Download PDF from URL and save to local file"""
    # Convert http to https for cal-access.sos.ca.gov
    if url.startswith("http://cal-access.sos.ca.gov"):
        url = url.replace("http://", "https://")
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

def analyze_with_gemini(pdf_path: str, api_key: str) -> str:
    """Upload PDF to Gemini and get JSON response"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = """Extract all bill numbers (like AB 1049, SB 1034) mentioned in the text/document. 
    Return the results in a specific JSON format with one field:
    
    An 'interests' array containing the bill numbers as strings
    
    Rules:
    - Include all Assembly Bills (AB) and Senate Bills (SB)
    - Maintain the exact format of the bill numbers (e.g., "AB 1049" not "AB1049")
    - Include each bill only once, even if mentioned multiple times
    - If no bills are found, return an empty array
    
    Expected JSON Format:
    {
        "interests": ["AB 1049", "SB 1034", "AB 45"]
    }"""
    
    pdf_file = genai.upload_file(pdf_path)
    response = model.generate_content([prompt, pdf_file])
    return response.text

def process_company_filings(csv_path: str, pdf_dir: str, api_key: str):
    """Process company filings with retries for failed PDF downloads"""
    logging.info(f"Starting processing: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create PDF directory if it doesn't exist
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Process each filing
    for idx, row in df.iterrows():
        try:
            # Add delay between requests
            sleep(uniform(1.0, 2.0))
            
            pdf_url = row['filing_url']
            filename = f"document_{idx}.pdf"
            pdf_path = os.path.join(pdf_dir, filename)
            
            # Download and analyze
            download_pdf(pdf_url, pdf_path)
            result = analyze_with_gemini(pdf_path, api_key)
            
            # Store result
            df.at[idx, 'gemini_analysis'] = result
            
            # Save progress
            df.to_csv(csv_path.replace('.csv', '_processed.csv'), index=False)
            logging.info(f"Successfully processed row {idx}")
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {str(e)}")
            df.at[idx, 'gemini_analysis'] = f"Error: {str(e)}"
            df.to_csv(csv_path.replace('.csv', '_processed.csv'), index=False)

    return df

def extract_unique_bills(df: pd.DataFrame) -> List[str]:
    """Extract all unique bill numbers from the Gemini analysis"""
    all_interests = []
    for _, row in df.iterrows():
        try:
            # Clean up JSON formatting if needed
            analysis = row['gemini_analysis'].replace("```json", "").replace("```", "")
            interests = json.loads(analysis)['interests']
            all_interests.extend(interests)
        except Exception as e:
            logging.error(f"Error parsing interests: {e}")
            
    return sorted(list(set(all_interests)))

def main():
    # Configuration
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    csv_path = "company_filings.csv"  # Your input CSV with filing URLs
    pdf_dir = "pdfs"
    
    try:
        # Process all filings
        df = process_company_filings(csv_path, pdf_dir, api_key)
        
        # Extract unique bills
        unique_bills = extract_unique_bills(df)
        print(f"Found {len(unique_bills)} unique bills")
        print("Bills:", unique_bills)
        
        logging.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Critical error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()