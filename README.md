# California Tech Industry Legislative Analysis

## Purpose
This project aims to provide deep insights into how the technology industry engages with and influences California's legislative process. By analyzing lobbying activities, bill content, and industry spending patterns, it reveals key policy priorities and legislative trends shaping tech regulation in California.

Key insights provided:
- Understanding which bills attract the most industry attention and spending
- Identifying patterns in legislative focus areas through semantic clustering
- Tracking company-specific lobbying priorities and strategies
- Analyzing the evolution of tech policy approaches
- Mapping relationships between different legislative initiatives

## Features
- **Automated Bill Analysis**: Uses Google's Gemini AI to generate detailed bill summaries and extract key themes
- **Semantic Clustering**: Groups related bills using OpenAI embeddings to identify legislative patterns
- **Interactive Dashboard**: Visualizes lobbying activities, spending patterns, and bill relationships
- **Company-Specific Insights**: Tracks individual company priorities and spending
- **Legislative Trend Analysis**: Maps the evolution of tech policy approaches

## Components

### 1. Data Processing Pipeline

#### Bill Summarization (`ca_bill_summarization.py`)
- Processes legislative bills using Google's Gemini AI
- Generates structured summaries and topic tags
- Implements batch processing with error handling

#### Embeddings Generation (`ca_bills_embedding.py`)
- Creates embeddings for bill summaries using OpenAI API
- Implements batched processing with rate limiting

#### Clustering Analysis (`cluster_summarization.py`)
- Performs K-means clustering on bill embeddings
- Generates cluster summaries and analysis
- Extracts key themes and policy patterns

#### Lobbying Data Processing (`get_lobbying_data.py`)
- Extracts bill references from lobbying documents
- Processes PDF filings and tracks spending

### 2. Interactive Dashboard (`main.py`)
Built with Streamlit, provides:
- Overview of tech industry lobbying activities
- Interactive 3D cluster visualization
- Company-specific analysis
- Bill relationship exploration

## Setup

### Prerequisites
```bash
# Python 3.8+ required
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### Running the Pipeline
```bash
python ca_bill_summarization.py
python ca_bills_embedding.py
python cluster_summarization.py
python get_lobbying_data.py
```

### Launch Dashboard
```bash
streamlit run main.py
```

## Dependencies
- `streamlit`: Dashboard framework
- `google.generativeai`: Gemini API
- `openai`: OpenAI API
- `pandas`: Data processing
- `scikit-learn`: Clustering and analysis
- `plotly`: Interactive visualizations
- `numpy`: Numerical operations

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Google Gemini AI for bill summarization
- OpenAI for embedding generation
- California Legislative Information System for bill data