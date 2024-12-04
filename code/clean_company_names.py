import pandas as pd
import re

def standardize_company_name(name: str) -> str:
    """Standardize company names by cleaning and mapping to canonical names"""
    if pd.isna(name):
        return name
        
    name = name.strip().upper()
    
    # Define mapping for known variants
    company_mapping = {
        # Apple
        re.compile(r'APPLE\s*,?\s*INC\.?'): 'APPLE INC.',
        
        # Google (handle complex cases first)
        re.compile(r'RXN GROUP OBO GOOGLE.*'): 'GOOGLE LLC',
        re.compile(r'KP PUBLIC AFFAIRS OBO.*GOOGLE.*'): 'GOOGLE LLC',
        re.compile(r'GOOGLE\s*,?\s*LLC.*GOOGLE CLIENT SERVICES.*'): 'GOOGLE LLC',
        
        # Meta
        re.compile(r'META\s+PLATFORMS\s*,?\s*INC\.?'): 'META PLATFORMS INC.',
        
        # Microsoft
        re.compile(r'MICROSOFT\s+CORPORATION'): 'MICROSOFT CORPORATION',
        
        # NVCA
        re.compile(r'NATIONAL\s+VENTURE\s+CAPITAL\s+ASSOCIATION'): 'NATIONAL VENTURE CAPITAL ASSOCIATION',
        
        # NetChoice
        re.compile(r'NETCHOICE'): 'NETCHOICE',
        
        # OpenAI
        re.compile(r'OPENAI\s+O[PC]{2}O\s*,?\s*LLC'): 'OPENAI INC.',
        
        # Y Combinator
        re.compile(r'Y\s+COMBINATOR\s+MANAGEMENT\s*,?\s*LLC'): 'Y COMBINATOR MANAGEMENT LLC'
    }
    
    # Apply mappings
    for pattern, replacement in company_mapping.items():
        if pattern.search(name):
            return replacement
            
    return name

# Read the CSV
df = pd.read_csv('combined_lobby_activity_processed_retried.csv')

# Store original company names for verification
original_names = df['employer'].unique()

# Clean company names
df['employer'] = df['employer'].apply(standardize_company_name)

# Show changes made
changes = pd.DataFrame({
    'Original': sorted(original_names),
    'Cleaned': [standardize_company_name(name) for name in sorted(original_names)]
})
print("Name Changes Made:")
for _, row in changes.iterrows():
    if row['Original'] != row['Cleaned']:
        print(f"{row['Original']} -> {row['Cleaned']}")

# Save updated CSV
df.to_csv('combined_lobby_activity_processed_cleaned.csv', index=False)