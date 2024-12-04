import requests
import json
import base64
import re
import pandas as pd
from typing import Dict, Optional, Tuple, List
import time
from datetime import datetime
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

def extract_bill_data(bill_details: Dict, bill_text: str) -> Dict:
    """Extract relevant fields from bill details and text"""
    bill = bill_details['bill']
    
    # Get final votes
    votes = bill.get('votes', [])
    final_assembly_vote = None
    final_senate_vote = None
    
    for vote in reversed(votes):
        if not final_assembly_vote and vote['chamber'] == 'A':
            final_assembly_vote = f"{vote['yea']}-{vote['nay']}"
        if not final_senate_vote and vote['chamber'] == 'S':
            final_senate_vote = f"{vote['yea']}-{vote['nay']}"
        if final_assembly_vote and final_senate_vote:
            break
    
    # Get sponsors
    sponsors = bill.get('sponsors', [])
    primary_sponsor = sponsors[0] if sponsors else {}
    
    # Get text versions
    texts = bill.get('texts', [])
    latest_text = texts[-1] if texts else {}
    
    # Get first and last actions
    history = bill.get('history', [])
    first_action = history[0] if history else {}
    last_action = history[-1] if history else {}
    
    return {
        'bill_number': bill.get('bill_number'),
        'title': bill.get('title'),
        'description': bill.get('description'),
        'status_date': bill.get('status_date'),
        'status': bill.get('status'),
        'url': bill.get('url'),
        'state_link': bill.get('state_link'),
        
        'introduced_date': first_action.get('date'),
        'last_action_date': last_action.get('date'),
        'last_action': last_action.get('action'),
        
        'primary_sponsor': primary_sponsor.get('name'),
        'primary_sponsor_party': primary_sponsor.get('party'),
        'cosponsor_count': len(sponsors) - 1 if sponsors else 0,
        'all_sponsors': '; '.join([s.get('name', '') for s in sponsors]),
        
        'final_assembly_vote': final_assembly_vote,
        'final_senate_vote': final_senate_vote,
        'total_votes_count': len(votes),
        
        'latest_text_date': latest_text.get('date'),
        'latest_text_type': latest_text.get('type'),
        'latest_text_url': latest_text.get('url'),
        'text_versions_count': len(texts),
        
        'full_text': bill_text,
        'text_length': len(bill_text) if bill_text else 0
    }

class LegiScanAPI:
    def __init__(self, api_key: str):
        """Initialize LegiScan API client with connection pooling"""
        self.api_key = api_key
        self.base_url = "https://api.legiscan.com/"
        self.rate_limit_delay = 0.5  # Reduced delay between API calls
        self.session = None
        
    async def _init_session(self):
        """Initialize aiohttp session for connection pooling"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_request(self, params: Dict) -> Dict:
        """Make async API request with error handling and rate limiting"""
        await self._init_session()
        params['key'] = self.api_key
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Async rate limiting
            async with self.session.get(self.base_url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            print(f"Error making request: {e}")
            return {}

    async def get_bill_by_number_async(self, bill_number: str, state: str = 'CA') -> Tuple[Optional[Dict], str]:
        """Async version of get_bill_by_number"""
        search_number = re.sub(r'([A-Za-z])(\d)', r'\1 \2', bill_number.upper())
        formatted_number = search_number.replace(" ", "")
        
        search_results = await self._make_request({
            'op': 'getSearch',
            'state': state,
            'query': search_number
        })
        
        if search_results.get('status') != 'OK':
            return None, "Search failed"
            
        searchresult = search_results['searchresult']
        for key, bill in searchresult.items():
            if isinstance(bill, dict) and key != 'summary':
                current_bill_number = bill.get('bill_number', '').upper()
                if current_bill_number == formatted_number:
                    bill_details = await self._make_request({
                        'op': 'getBill',
                        'id': bill['bill_id']
                    })
                    return bill_details, ""
                
        return None, f"Bill {bill_number} not found"

    async def get_bill_text_content_async(self, bill_details: Dict) -> Tuple[Optional[str], str]:
        """Async version of get_bill_text_content"""
        if not bill_details or bill_details.get('status') != 'OK':
            return None, "Invalid bill details"
            
        texts = bill_details['bill']['texts']
        if not texts:
            return None, "No text versions available"
            
        latest_text = texts[-1]
        text_data = await self._make_request({
            'op': 'getBillText',
            'id': latest_text['doc_id']
        })
        
        if text_data.get('status') != 'OK':
            return None, "Failed to get bill text"
            
        try:
            decoded_text = base64.b64decode(text_data['text']['doc']).decode('utf-8')
            return decoded_text, ""
        except Exception as e:
            return None, f"Error decoding text: {str(e)}"

class BatchBillDataCollector:
    def __init__(self, api_key: str, batch_size: int = 10, save_interval: int = 50, filename: str = 'ca_bills_data.csv'):
        """
        Initialize collector with batching capabilities
        batch_size: Number of bills to process concurrently
        save_interval: Number of bills to process before saving progress
        filename: Name of the CSV file to save to
        """
        self.api_key = api_key
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.filename = filename
        self.backup_filename = f'{filename}.backup'
        self.legiscan = LegiScanAPI(api_key)
        self.progress_file = f'{filename}.progress'
        
    def load_progress(self) -> Tuple[set, List[Dict], int]:
        """Load progress from files"""
        try:
            df = pd.read_csv(self.filename)
            processed_bills = set(df['bill_number'].values)
            
            # Load last processed index
            try:
                with open(self.progress_file, 'r') as f:
                    last_index = int(f.read().strip())
            except:
                last_index = 0
                
            return processed_bills, df.to_dict('records'), last_index
        except FileNotFoundError:
            return set(), [], 0

    def save_progress(self, bills_data: List[Dict], current_index: int):
        """Save current data and progress"""
        df = pd.DataFrame(bills_data)
        
        # Save to backup first
        df.to_csv(self.backup_filename, index=False)
        
        # Then save to main file
        df.to_csv(self.filename, index=False)
        
        # Save progress index
        with open(self.progress_file, 'w') as f:
            f.write(str(current_index))
        
        print(f"Progress saved - {len(bills_data)} bills processed, current index: {current_index}")

    async def process_bill_batch(self, bill_numbers: List[str]) -> List[Dict]:
        """Process a batch of bills concurrently"""
        tasks = []
        for bill_number in bill_numbers:
            tasks.append(self.process_single_bill(bill_number))
        return await asyncio.gather(*tasks)

    async def process_single_bill(self, bill_number: str) -> Dict:
        """Process a single bill asynchronously"""
        try:
            bill_details, error = await self.legiscan.get_bill_by_number_async(bill_number)
            if error:
                print(f"Error getting bill details for {bill_number}: {error}")
                return None
                
            bill_text, error = await self.legiscan.get_bill_text_content_async(bill_details)
            if error:
                print(f"Error getting bill text for {bill_number}: {error}")
                bill_text = None
                
            return extract_bill_data(bill_details, bill_text)
            
        except Exception as e:
            print(f"Error processing bill {bill_number}: {e}")
            return None

    async def build_bills_dataframe(self, bill_numbers: List[str]) -> pd.DataFrame:
        """Build dataframe from list of bill numbers using batch processing"""
        processed_bills, existing_data, last_index = self.load_progress()
        bills_data = existing_data
        
        if processed_bills:
            print(f"Resuming from previous run - {len(processed_bills)} bills already processed")
        
        # Filter out already processed bills and start from last index
        remaining_bills = [b for b in bill_numbers[last_index:] if b not in processed_bills]
        total_remaining = len(remaining_bills)
        print(f"Processing {total_remaining} remaining bills in batches of {self.batch_size}...")
        
        try:
            # Process bills in batches
            for i in range(0, len(remaining_bills), self.batch_size):
                batch = remaining_bills[i:i + self.batch_size]
                print(f"\nProcessing batch {i//self.batch_size + 1}, bills {i+1}-{i+len(batch)}")
                
                batch_results = await self.process_bill_batch(batch)
                valid_results = [r for r in batch_results if r is not None]
                bills_data.extend(valid_results)
                
                current_index = last_index + i + len(batch)
                
                # Save at intervals
                if len(bills_data) % self.save_interval == 0:
                    self.save_progress(bills_data, current_index)
                    
        except Exception as e:
            print(f"Error during batch processing: {e}")
            # Save progress on error
            if len(bills_data) > len(processed_bills):
                self.save_progress(bills_data, current_index)
                
        finally:
            # Close the API session
            await self.legiscan._close_session()
            
        # Final save
        if len(bills_data) > len(processed_bills):
            self.save_progress(bills_data, len(bill_numbers))
        
        return pd.DataFrame(bills_data)

async def main_async(bill_numbers: List[str], api_key: str, filename: str = 'ca_bills_data.csv', batch_size: int = 10):
    """Async main function to run the batch bill data collection"""
    collector = BatchBillDataCollector(api_key, batch_size=batch_size, filename=filename)
    df = await collector.build_bills_dataframe(bill_numbers)
    print("\nData collection completed!")
    print(f"Total bills processed: {len(df)}")
    return df

def main(bill_numbers: List[str], api_key: str, filename: str = 'ca_bills_data.csv', batch_size: int = 10):
    """Jupyter-compatible wrapper for the async main function"""
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async function in the current notebook's event loop
    return loop.run_until_complete(main_async(bill_numbers, api_key, filename, batch_size))

if __name__ == "__main__":
    # Read and process the CSV to get unique interests
    df = pd.read_csv("combined_lobby_activity_processed_retried.csv")
    df['gemini_analysis'] = df['gemini_analysis'].replace("```json ", "").replace("```", "")
    
    # Process each row's gemini_analysis
    for i, row in df.iterrows():
        try:
            df.at[i, 'gemini_analysis'] = df.at[i, 'gemini_analysis'].replace("```json", "").replace("```", "")
            df.at[i, 'gemini_analysis'] = json.loads(df.at[i, 'gemini_analysis'])
        except json.JSONDecodeError:
            print(f"Error parsing JSON for row {i}")
            df.at[i, 'gemini_analysis'] = df.at[i, 'gemini_analysis']
    
    # Extract all interests
    all_interests = []
    for i, row in df.iterrows():
        try:
            all_interests.extend(row['gemini_analysis']['interests'])
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Error extracting interests for row {i}")
    
    # Get unique interests
    unique_interests = sorted(list(set(all_interests)))
    
    # Process the bills
    API_KEY = "7f6737cd6d4991d39a80ed8bc5888d2e"
    df = main(unique_interests, API_KEY, batch_size=10)