import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import logging
from typing import List, Dict
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from typing import List, Dict, Any
import json
from sklearn.metrics import silhouette_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bill_clustering.log'),
        logging.StreamHandler()
    ]
)

def setup_gemini(api_key: str) -> genai.GenerativeModel:
    """Configure and return Gemini model"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def parse_embedding(embedding_str: str) -> np.ndarray:
    """Parse embedding string to numpy array"""
    try:
        return np.array(ast.literal_eval(embedding_str))
    except Exception as e:
        logging.error(f"Error parsing embedding: {str(e)}")
        return None

def find_optimal_clusters(data: np.ndarray, max_clusters: int = 10) -> int:
    """
    Find optimal number of clusters using elbow method and silhouette score
    """
    distortions = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)
    
    for k in K:
        logging.info(f"Testing k={k} clusters")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Calculate the rate of change in distortion
    distortion_changes = np.diff(distortions)
    rate_of_change = np.diff(distortion_changes)
    
    # Find the elbow point (where the rate of change starts to level off)
    elbow_point = np.argmin(rate_of_change) + 2
    
    # Find the point with the highest silhouette score
    silhouette_point = np.argmax(silhouette_scores) + 2
    
    # Take the average of elbow point and silhouette point
    optimal_clusters = min(max(2, round((elbow_point + silhouette_point) / 2)), max_clusters)
    
    logging.info(f"Optimal number of clusters determined: {optimal_clusters}")
    return optimal_clusters

def get_cluster_keywords(df: pd.DataFrame, cluster_label: int, n_keywords: int = 10) -> List[str]:
    """Extract distinctive keywords for a cluster using TF-IDF"""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    # Use gemini_summary for keyword extraction
    cluster_texts = df[df['cluster'] == cluster_label]['gemini_summary'].fillna('').tolist()
    other_texts = df[df['cluster'] != cluster_label]['gemini_summary'].fillna('').tolist()
    
    # Get TF-IDF scores for both clusters
    all_texts = cluster_texts + other_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate average TF-IDF scores for cluster vs. non-cluster
    cluster_avg = tfidf_matrix[:len(cluster_texts)].mean(axis=0).A1
    other_avg = tfidf_matrix[len(cluster_texts):].mean(axis=0).A1
    
    # Get distinctive terms (high in cluster, low in others)
    distinctiveness = cluster_avg - other_avg
    top_indices = distinctiveness.argsort()[-n_keywords:][::-1]
    
    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[i] for i in top_indices]

def get_cluster_stats(df: pd.DataFrame, cluster_label: int) -> Dict:
    """Get detailed statistics for a cluster"""
    cluster_df = df[df['cluster'] == cluster_label]
    
    # Calculate vote statistics
    vote_stats = {
        'assembly_votes': cluster_df['final_assembly_vote'].dropna().tolist(),
        'senate_votes': cluster_df['final_senate_vote'].dropna().tolist()
    }
    
    stats = {
        'size': len(cluster_df),
        'sponsor_parties': Counter(cluster_df['primary_sponsor_party'].fillna('Unknown')).most_common(),
        'status_distribution': Counter(cluster_df['status'].fillna('Unknown')).most_common(),
        'avg_cosponsor_count': cluster_df['cosponsor_count'].mean(),
        'vote_stats': vote_stats,
        'date_range': {
            'earliest': cluster_df['introduced_date'].min(),
            'latest': cluster_df['last_action_date'].max()
        }
    }
    
    return stats

def summarize_cluster(
    model: genai.GenerativeModel,
    df: pd.DataFrame,
    cluster_label: int,
    keywords: List[str]
) -> str:
    """Generate a summary for a cluster using Gemini"""
    cluster_df = df[df['cluster'] == cluster_label]
    
    # Get top 25 most representative bills (using ones with most cosponsors as a proxy for importance)
    sample_bills = cluster_df.nlargest(25, 'cosponsor_count')[
        ['bill_number', 'title', 'gemini_summary', 'status', 'cosponsor_count']
    ].to_dict('records')
    
    sample_text = "\n".join([
        f"Bill {b['bill_number']} ({b['status']}, {b['cosponsor_count']} cosponsors): {b['title']}\nSummary: {b['gemini_summary'][:200]}..."
        for b in sample_bills
    ])
    
    prompt = f"""
    Analyze this cluster of {len(cluster_df)} related bills. Key themes: {', '.join(keywords)}

    Sample bills (25 most significant):
    {sample_text}

    Provide a comprehensive, information-dense analysis that covers:
    1. Core thematic elements and policy objectives (be specific about policy mechanisms)
    2. Progression of policy approaches across the bills
    3. Key variations in approach or scope
    4. Common regulatory or implementation strategies
    5. Critical policy implications and potential impacts

    Focus on concrete details and patterns. Prioritize specificity over general statements. Include actual examples from the bills to support key points.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating summary for cluster {cluster_label}: {str(e)}")
        return f"Error generating summary: {str(e)}"

def main():
    # Configuration
    INPUT_FILE = "ca_bills_summarized_with_embeddings.csv"
    OUTPUT_FILE = "clustered_bills_analysis.txt"
    OUTPUT_CSV = "ca_bills_clustered.csv"
    MAX_CLUSTERS = 10
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    logging.info("Starting bill clustering and analysis process")
    
    try:
        # Read data
        df = pd.read_csv(INPUT_FILE)
        logging.info(f"Loaded {len(df)} bills")
        
        # Parse embeddings
        embeddings = np.vstack(df['summary_embedding'].apply(parse_embedding))
        
        # Normalize embeddings
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        
        # Find optimal number of clusters
        n_clusters = find_optimal_clusters(normalized_embeddings, MAX_CLUSTERS)
        logging.info(f"Using {n_clusters} clusters based on optimization")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(normalized_embeddings)
        df['cluster'] = clusters
        
        # Save updated CSV with cluster assignments
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Saved clustered data to {OUTPUT_CSV}")
        
        # Set up Gemini
        model = setup_gemini(api_key)
        
        # Analyze each cluster
        cluster_analyses = []
        for cluster_label in range(n_clusters):
            logging.info(f"Analyzing cluster {cluster_label}")
            
            # Get cluster keywords
            keywords = get_cluster_keywords(df, cluster_label)
            
            # Get cluster statistics
            stats = get_cluster_stats(df, cluster_label)
            
            # Get cluster summary
            summary = summarize_cluster(model, df, cluster_label, keywords)
            
            cluster_analyses.append({
                'cluster': cluster_label,
                'size': stats['size'],
                'keywords': keywords,
                'summary': summary,
                'stats': stats
            })
        
        # Save results
        with open(OUTPUT_FILE, 'w') as f:
            f.write("California Bills Cluster Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of clusters determined: {n_clusters}\n\n")
            
            for analysis in cluster_analyses:
                f.write(f"Cluster {analysis['cluster']}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Size: {analysis['size']} bills\n")
                f.write(f"Date Range: {analysis['stats']['date_range']['earliest']} to {analysis['stats']['date_range']['latest']}\n")
                f.write(f"Key themes: {', '.join(analysis['keywords'])}\n\n")
                
                # Write detailed statistics
                f.write("Cluster Statistics:\n")
                f.write(f"- Party Distribution: {analysis['stats']['sponsor_parties']}\n")
                f.write(f"- Status Distribution: {analysis['stats']['status_distribution']}\n")
                f.write(f"- Average Cosponsors: {analysis['stats']['avg_cosponsor_count']:.1f}\n")
                f.write(f"- Vote Information: {analysis['stats']['vote_stats']}\n\n")
                
                f.write("Thematic Analysis:\n")
                f.write(analysis['summary'])
                f.write("\n\n" + "=" * 50 + "\n\n")
        
        logging.info(f"Analysis complete. Results saved to {OUTPUT_FILE}")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()