import streamlit as st
import pandas as pd
import json
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import ast
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="CA Tech Lobbying Analysis",
    page_icon="📊",
    layout="wide"
)

# Constants
COMMON_SUFFIXES = ['Inc.', 'LLC', 'Corporation', 'Corp.', 'Inc', 'Ltd.', 'Limited', ',']

# Top bills data (remains the same)
TOP_BILLS_INFO = {
    "SB 1047": {
        "title": "Safe and Secure Innovation for Frontier AI Models Act",
        "spending": 166575.38,
        "summary": """Regulates powerful AI models based on computing power and training costs. 
        Requires safety protocols, audits, and incident reporting. Establishes Board of Frontier 
        Models and CalCompute public cloud computing cluster.""",
        "companies": ["Google", "Meta", "Microsoft", "NVCA", "OpenAI", "Y Combinator"],
        "key_points": ["Safety protocols", "Regular audits", "Oversight board", "Public computing infrastructure"]
    },
    "AB 2930": {
        "title": "Automated Decision Systems Impact Assessment",
        "spending": 117059.23,
        "summary": """Mandates impact assessments for automated decision systems, focusing on 
        algorithmic discrimination risk. Requires notification and opt-out options for affected 
        individuals.""",
        "companies": ["Google", "Meta", "Microsoft", "NetChoice", "OpenAI"],
        "key_points": ["Impact assessments", "Discrimination prevention", "Consumer rights", "Civil penalties"]
    },
    "AB 3211": {
        "title": "Digital Content Provenance Standards",
        "spending": 111337.63,
        "summary": """Requires provenance data for AI-generated content, detection tools, and 
        adversarial testing. Mandates platform transparency reports on synthetic content.""",
        "companies": ["Google", "Meta", "Microsoft", "OpenAI"],
        "key_points": ["Content provenance", "Detection tools", "Platform requirements", "Transparency reporting"]
    },
    "SB 942": {
        "title": "AI Transparency Act",
        "spending": 111337.63,
        "summary": """Requires AI detection tools and content disclosure options for large GenAI 
        providers. Mandates both manifest and latent disclosures.""",
        "companies": ["Google", "Meta", "Microsoft", "OpenAI"],
        "key_points": ["AI detection", "Content disclosure", "User transparency", "Civil penalties"]
    },
    "AB 2013": {
        "title": "AI Training Data Transparency",
        "spending": 111337.63,
        "summary": """Mandates public disclosure of AI training datasets, including sources, 
        purposes, and data characteristics.""",
        "companies": ["Google", "Meta", "Microsoft", "OpenAI"],
        "key_points": ["Data transparency", "Dataset documentation", "Public disclosure", "Training data details"]
    }
}

def clean_employer_name(name):
    """Normalize employer names by removing common suffixes and extra spaces."""
    if pd.isna(name):
        return ""
    name = name.strip()
    # Remove common suffixes
    for suffix in COMMON_SUFFIXES:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    # Remove any trailing commas or periods
    name = re.sub(r'[.,]$', '', name)
    return name

def show_introduction(company_data):
    """Display the introduction section with key metrics and overview"""
    st.title("California Tech Industry Lobbying Analysis 2023-2024 📊")
    
    st.markdown("""
    This dashboard provides insights into lobbying activities by major technology companies 
    in California during the 2023-2024 legislative session. Track spending, analyze focused 
    bills, and understand the key policy areas that shape the tech industry's legislative priorities.
    """)
    
    # Add tech industry priorities analysis
    st.header("Key Tech Industry Legislative Priorities")
    
    st.markdown("""
    Analysis of the top bills by lobbying spend reveals clear priorities for the tech industry:
    
    1. **AI Safety and Regulation**: The highest-spend bill (SB 1047) focuses on frontier AI 
    model safety and oversight, indicating the industry's engagement with AI governance.
    
    2. **Algorithmic Accountability**: Significant focus on automated decision systems and 
    their impact on consumers, with emphasis on preventing discrimination.
    
    3. **Content Authentication**: Multiple high-priority bills address AI-generated content 
    verification and provenance, showing industry involvement in addressing synthetic media concerns.
    
    4. **Transparency Requirements**: Consistent theme across bills of establishing clear 
    disclosure requirements for AI systems, training data, and generated content.
    
    5. **Infrastructure Development**: Interest in public computing resources (CalCompute) 
    suggests industry engagement in public AI infrastructure development.
    """)
    
    # Calculate key metrics for tech companies
    tech_total_spending = sum(
        data['total_spending'] for data in company_data.values()
    )
    
    tech_total_bills = len(set().union(*[
        data['bills'] for data in company_data.values()
    ]))
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Total Tech Industry Lobbying Spending",
            f"${tech_total_spending:,.2f}"
        )
    
    with col2:
        st.metric(
            "Unique Bills Lobbied",
            tech_total_bills
        )
    
    # Create spending comparison chart
    spending_data = {
        company: data['total_spending']
        for company, data in company_data.items()
    }
    # Sort spending data by values in descending order
    spending_data_sorted = dict(sorted(spending_data.items(), key=lambda x: x[1], reverse=True))
    
    spending_df = pd.DataFrame({
        'Company': list(spending_data_sorted.keys()),
        'Spending': list(spending_data_sorted.values())
    })
    
    fig = px.bar(
        spending_df,
        x='Company',
        y='Spending',
        title="Lobbying Spending by Tech Company",
        labels={'Spending': 'Total Spending ($)'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat='$,.0f'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add top bills analysis section
    st.header("Top Bills by Lobbying Spend")
    
    # Create tabs for each top bill
    bill_tabs = st.tabs(TOP_BILLS_INFO.keys())
    
    for tab, (bill_id, info) in zip(bill_tabs, TOP_BILLS_INFO.items()):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(info["title"])
                st.markdown(f"**Total Spending:** ${info['spending']:,.2f}")
                st.markdown("**Summary:**")
                st.markdown(info["summary"])
                
            with col2:
                st.markdown("**Key Points:**")
                for point in info["key_points"]:
                    st.markdown(f"- {point}")
                
                st.markdown("**Companies Involved:**")
                for company in info["companies"]:
                    st.markdown(f"- {company}")

def show_bill_overview(bill_data):
    """Display the bill overview section"""
    st.header("Bills Overview")
    
    # Convert bill data to a list and sort by total spending
    bill_list = [
        {
            'bill_number': bill,
            'total_spending': data['total_spending'],
            'summary': data['summary'],
            'companies': sorted(data['companies']),
            'tags': data['tags']
        }
        for bill, data in bill_data.items()
    ]
    
    sorted_bills = sorted(bill_list, key=lambda x: x['total_spending'], reverse=True)
    
    # Create DataFrame for display
    df_display = pd.DataFrame(sorted_bills)
    
    # Format the data for display
    df_display['total_spending'] = df_display['total_spending'].apply(lambda x: f"${x:,.2f}")
    df_display['companies'] = df_display['companies'].apply(lambda x: ', '.join(x))
    df_display['tags'] = df_display['tags'].apply(lambda x: ', '.join(x))
    
    # Rename columns for display
    df_display.columns = ['Bill Number', 'Total Spending', 'Summary', 'Companies', 'Tags']
    
    # Display as interactive table
    st.dataframe(
        df_display,
        column_config={
            'Summary': st.column_config.TextColumn(width="large"),
            'Companies': st.column_config.TextColumn(width="medium"),
            'Tags': st.column_config.TextColumn(width="medium")
        },
        hide_index=True
    )

def clean_bill_number(bill_num: str) -> str:
    """Normalize bill numbers by removing spaces"""
    if pd.isna(bill_num):
        return ""
    return re.sub(r'\s+', '', str(bill_num))

def parse_json_field(json_str: str) -> list:
    """Parse JSON fields, handling various formats"""
    if pd.isna(json_str):
        return []
        
    try:
        # If it's already a list, return it
        if isinstance(json_str, list):
            return json_str
            
        # Clean up the string by removing code block markers and whitespace
        if isinstance(json_str, str):
            cleaned = json_str.replace('```json', '').replace('```', '').strip()
            
            # If the cleaned string looks like a list, parse it directly
            if cleaned.startswith('[') and cleaned.endswith(']'):
                return json.loads(cleaned)
                
            # Otherwise try parsing as a JSON object with 'interests' key
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data.get('interests', [])
                
            return []
    except Exception as e:
        st.write(f"Error parsing: {json_str[:100]}...")
        st.write(f"Error: {str(e)}")
        return []

@st.cache_data
def load_data():
    """Load and process the data files"""
    # Read CSV files
    lobby_df = pd.read_csv('data/lobbying_activity.csv')
    bills_df = pd.read_csv('data/ca_bills_embedded_and_clustered.csv')  # Using clustered version

    # Process embeddings for 3D visualization
    bills_df['embedding'] = bills_df['summary_embedding'].apply(lambda x: np.array(ast.literal_eval(x)) if pd.notna(x) else None)
    
    # Remove rows with None embeddings
    bills_df = bills_df[bills_df['embedding'].notna()]
    
    # Convert embeddings to numpy array for PCA
    embeddings_array = np.vstack(bills_df['embedding'])
    
    # Perform PCA for 3D visualization
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(embeddings_array)
    
    # Add 3D coordinates to bills_df
    bills_df['x'] = coords_3d[:, 0]
    bills_df['y'] = coords_3d[:, 1]
    bills_df['z'] = coords_3d[:, 2]

    # Process bills data
    bills_df['bill_number_clean'] = bills_df['bill_number'].apply(clean_bill_number)
    bills_df['tags'] = bills_df['gemini_topics'].apply(parse_json_field)
    
    # Create bills lookup
    bills_lookup = {}
    for _, row in bills_df.iterrows():
        clean_num = row['bill_number_clean']
        if clean_num:
            bills_lookup[clean_num] = {
                'gemini_summary': row['gemini_summary'],
                'tags': row['tags'],
                'cluster': row['cluster'] if 'cluster' in row else None,
                'x': row['x'],
                'y': row['y'],
                'z': row['z']
            }

    # Process lobbying data
    lobby_df['interests'] = lobby_df['gemini_analysis'].apply(parse_json_field)
    
    # Normalize employer names
    lobby_df['employer_clean'] = lobby_df['employer'].apply(clean_employer_name)
    
    # Get the list of unique companies after normalization
    unique_companies = lobby_df['employer_clean'].dropna().unique()
    
    # Aggregate by company
    company_data = {}
    for _, row in lobby_df.iterrows():
        company = row['employer_clean']
        if pd.isna(company) or company == "":
            continue
            
        if company not in company_data:
            company_data[company] = {
                'total_spending': 0,
                'bills': set(),
                'all_tags': [],
                'quarterly_spending': []
            }
            
        # Add spending
        compensation = row['compensation'] if not pd.isna(row['compensation']) else 0
        company_data[company]['total_spending'] += compensation
        
        # Add quarterly spending
        if not pd.isna(row['start_date']) and not pd.isna(row['compensation']):
            company_data[company]['quarterly_spending'].append({
                'date': pd.to_datetime(row['start_date']),
                'amount': row['compensation']
            })
        
        # Add bills and their tags
        for bill in row['interests']:
            clean_bill = clean_bill_number(bill)
            if clean_bill in bills_lookup:
                company_data[company]['bills'].add(bill)
                if isinstance(bills_lookup[clean_bill]['tags'], list):
                    company_data[company]['all_tags'].extend(bills_lookup[clean_bill]['tags'])

    # Aggregate data by bill
    bill_data = {}
    for company, data in company_data.items():
        bills_count = len(data['bills'])
        if bills_count == 0:
            continue
            
        spending_per_bill = data['total_spending'] / bills_count
        
        for bill in data['bills']:
            clean_bill = clean_bill_number(bill)
            if clean_bill not in bills_lookup:
                continue
                
            if bill not in bill_data:
                bill_data[bill] = {
                    'total_spending': 0,
                    'companies': set(),
                    'summary': bills_lookup[clean_bill]['gemini_summary'],
                    'tags': bills_lookup[clean_bill]['tags']
                }
            
            bill_data[bill]['total_spending'] += spending_per_bill
            bill_data[bill]['companies'].add(company)

    return company_data, bills_lookup, bill_data, unique_companies, bills_df

def show_cluster_visualization(bills_df):
    """Display 3D cluster visualization and detailed cluster analysis"""
    st.header("Bill Clusters 3D Visualization and Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This 3D visualization shows the relationships between bills based on their content embeddings. 
        Bills that are closer together in this space are more similar in content and theme. 
        Each color represents a different cluster of related bills.
        
        - **Hover** over points to see bill details
        - Use your mouse to **rotate** the visualization
        - **Scroll** to zoom in/out
        """)
        
        # Create the 3D scatter plot
        n_clusters = len(bills_df['cluster'].unique())
        colors = px.colors.qualitative.Set3[:n_clusters]
        
        # Create hover text
        bills_df['hover_text'] = bills_df.apply(
            lambda row: f"""
            <b>Bill {row['bill_number']}</b><br>
            <b>Title:</b> {row['title']}<br>
            <b>Status:</b> {row['status']}<br>
            <b>Cluster:</b> {row['cluster']}<br>
            <b>Summary:</b> {row['gemini_summary'][:200]}...<br>
            <b>Tags:</b> {', '.join(row['tags'])}
            """,
            axis=1
        )

        fig = go.Figure(data=[
            go.Scatter3d(
                x=bills_df[bills_df['cluster'] == c]['x'],
                y=bills_df[bills_df['cluster'] == c]['y'],
                z=bills_df[bills_df['cluster'] == c]['z'],
                mode='markers',
                name=f'Cluster {c}',
                marker=dict(
                    size=4,
                    color=colors[i],
                    opacity=0.8
                ),
                text=bills_df[bills_df['cluster'] == c]['hover_text'],
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family="Arial",
                    font_color='black'  # Set font color to black
                )
            ) for i, c in enumerate(sorted(bills_df['cluster'].unique()))
        ])

        # Update the layout
        fig.update_layout(
            height=800,
            showlegend=True,
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster Overview")
        
        cluster_map = {
            0: "Infrastructure & Environment",
            1: "Energy & Utilities",
            2: "Healthcare",
            3: "Waste & Product Safety",
            4: "Digital & Privacy",
            5: "Housing"
        }
        
        cluster_0_summary = """
                
        This analysis examines a cluster of 228 bills focusing on water, projects, tax, budget, wildfire, groundwater, boards, flood, drought, and government.  Based on the provided sample of 25 significant bills, several core thematic elements and policy objectives emerge.

        **1. Core Thematic Elements and Policy Objectives:**

        The dominant theme is **public finance and infrastructure investment**, specifically addressing environmental challenges and improving public services.  This manifests in several ways:

        * **Bond Acts for Infrastructure:**  Several bills propose massive bond acts to fund crucial infrastructure projects.  `SB867` ($10 billion) targets safe drinking water, wildfire prevention, drought preparedness, and clean air. `AB1567` proposes another large bond act addressing similar issues, adding flood protection, extreme heat mitigation, clean energy, and workforce development.  `AB408` focuses on climate-resilient farms and food access, while `SB638` specifically targets climate resiliency and flood protection. These bills utilize the policy mechanism of **debt financing** to address significant infrastructure needs.

        * **Taxation and Revenue Generation:**  Bills like `AB259` propose a wealth tax as a new revenue source, aiming to address wealth inequality and fund public services. This is a direct policy mechanism of **revenue enhancement** through a new tax structure.  Conversely, `ACA3` aims to alter constitutional appropriation limits, potentially impacting the state's ability to spend on public works.

        * **Local Government Finance Reform:** `ACA1` seeks to reform local government financing for affordable housing and public infrastructure, suggesting a shift in **intergovernmental fiscal relations** and potentially requiring voter approval for certain projects.  This demonstrates a move towards empowering localities while imposing checks and balances.

        * **Addressing Environmental Risks:**  Many bills explicitly address environmental threats.  `ACA16` proposes a constitutional amendment to enshrine environmental rights, representing a fundamental shift in **environmental policy**.  Other bills tackle specific issues like wildfire prevention (`SB867`, `AB1567`), drought preparedness (`SB867`, `AB1567`), and flood protection (`AB1567`, `SB638`). This points to a strategic approach towards proactive risk mitigation.


        **2. Progression of Policy Approaches:**

        The bills show a progression from addressing individual issues in isolation towards a more holistic, integrated approach.  Early bills may focus on specific problems (e.g., `SB650` on charitable raffles), while later proposals encompass broader strategies (e.g., `AB1567` combining multiple environmental and economic issues within a single bond act). This reflects a movement towards **comprehensive policy solutions** that address interconnected challenges.


        **3. Key Variations in Approach or Scope:**

        The bills vary significantly in scope and target:

        * **Scale of Intervention:** Some bills propose sweeping changes to the state constitution (`ACA1`, `ACA16`, `ACA3`, `ACA13`, `ACA11`), while others focus on more specific, targeted reforms (`AB421` on referendum measures, `AB1217` on business pandemic relief).

        * **Level of Government:** Some bills impact state-level policy and finance (`AB259`, `SB867`), while others affect local governments (`ACA1`, `AB341`).  Some empower tribal governments (`SB771`).

        * **Policy Mechanisms:**  As discussed earlier, mechanisms range from bond acts, tax increases, constitutional amendments, regulatory changes, and grant programs.

        * **Funding Sources:** Some bills rely on new taxes (`AB259`), while others use bond financing (`SB867`, `AB1567`). `AB513` proposes a grant program using existing state funds.


        **4. Common Regulatory or Implementation Strategies:**

        Several common implementation strategies are present:

        * **Bond Acts:**  This is a recurring theme, requiring voter approval and involving detailed allocation plans for funds.
        * **Constitutional Amendments:**  This demands supermajority approval and significant political mobilization.
        * **Regulatory Changes:**  Some bills modify existing laws, which requires detailed legislative drafting and consideration of potential unintended consequences.
        * **Grant Programs:**  These require the establishment of eligibility criteria, application processes, and oversight mechanisms.


        **5. Critical Policy Implications and Potential Impacts:**

        * **Fiscal Implications:** The large bond acts proposed have significant implications for the state's long-term debt. The wealth tax proposed in `AB259` could alter wealth distribution and potentially impact investment and economic activity.

        * **Environmental Impacts:**  The investment in water infrastructure, wildfire prevention, and climate resiliency can significantly impact California's environmental sustainability.  The success depends on effective implementation and resource allocation.

        * **Political Impacts:**  The proposed constitutional amendments could fundamentally change power dynamics within the state government. The wealth tax proposal is likely to generate significant political debate and opposition.

        * **Social Impacts:**  The investments in affordable housing, education, and workforce development can influence social equity and economic opportunity.

        In conclusion, the 228 bills reflect a multifaceted approach to addressing California's complex challenges.  The cluster suggests a strategic focus on infrastructure investment, environmental protection, and fiscal reform.  However, the success of these policies depends heavily on effective implementation, careful consideration of unintended consequences, and broad political support.  Further analysis would require examining the full text of all 228 bills and understanding the legislative processes and political contexts surrounding their introduction.

        """
        
        cluster_1_summary = """
        
Thematic Analysis:
## Analysis of 159 Related Bills: California Energy and Utility Policy

This analysis examines a cluster of 159 bills focused on energy, utilities, and related policy in California, based on the provided sample of 25 significant bills.  The core themes revolve around transitioning to cleaner energy sources, regulating utility practices, and managing the environmental impacts of energy production and consumption.

**1. Core Thematic Elements and Policy Objectives:**

The bills broadly address four key areas, using a variety of policy mechanisms:

* **Clean Energy Transition:**  Many bills aim to accelerate California's shift towards renewable energy sources (solar, wind, hydrogen) and reduce reliance on fossil fuels.  This is achieved through:
    * **Mandates:**  Bills like AB 841 (Industrial Heat Electrification Roadmap) and SB 1182 (Master Plan for Healthy, Sustainable, and Climate-Resilient Schools) mandate the development of plans and roadmaps for transitioning to cleaner energy.  SB 493 mandates assessing EV charging and hydrogen fueling infrastructure.
    * **Incentives:** SB 425 increases rebates for zero-emission electric pickup trucks, incentivizing their adoption. AB 2672 extends reduced energy rates to public housing, encouraging energy efficiency.
    * **Regulatory Changes:**  Bills like SB 286 streamline permitting processes for offshore wind projects, making them more economically viable. SB 1420 modifies CEQA streamlining for hydrogen production facilities.


* **Utility Regulation:**  Several bills directly address the regulation of public utilities (PUC) and their practices:
    * **Fixed Charges:** AB 1999 directly amends the Public Utilities Code to address electricity fixed charges, indicating a focus on the fairness and transparency of utility pricing structures.
    * **Environmental Review:** AB 914 aims to expedite the environmental review process for electrical infrastructure projects, potentially impacting the speed of infrastructure development.

* **Air Quality Improvement:** A significant subset of bills focuses on reducing vehicular and industrial air pollution:
    * **Fee Extensions:**  AB 126 and AB 241 extend increased smog abatement fees and vehicle registration fees, providing continued funding for clean transportation programs.  This illustrates a reliance on revenue generation for environmental initiatives.
    * **Emission Reduction:** AB 2083 mandates significant reductions in greenhouse gas emissions from industrial facilities, demonstrating a focus on industrial decarbonization.

* **Transportation Electrification:**  Several bills target the transition to electric vehicles:
    * **Public Agency Procurement:** AB 1594 mandates the procurement of medium- and heavy-duty zero-emission vehicles by public agencies, driving demand for electric vehicles in the public sector.


**2. Progression of Policy Approaches:**

The bills demonstrate a progression from primarily focusing on individual initiatives (e.g., extending fees, creating specific roadmaps) towards more comprehensive, system-wide approaches (e.g., Master Plans for schools and industrial heat electrification). This suggests a maturing policy landscape moving beyond isolated actions to integrated strategies.

**3. Key Variations in Approach or Scope:**

Significant variations exist in scope and approach:

* **Geographic Focus:**  Some bills focus on statewide impacts (e.g., AB 2083), while others target specific sectors (e.g., AB 1594, focusing on public agency utilities) or geographical areas (SB 286, focusing on offshore wind).
* **Policy Instruments:** The bills utilize a variety of policy instruments – mandates, incentives, fee extensions, regulatory changes – reflecting a multi-faceted approach to achieving policy objectives.
* **Time Horizons:**  Some bills involve immediate actions (e.g., fee extensions), while others lay out long-term strategies and roadmaps (e.g., the Industrial Heat Electrification Roadmap).  This demonstrates a combination of short-term fixes and long-term planning.


**4. Common Regulatory or Implementation Strategies:**

Several common strategies emerge:

* **Leveraging Existing Agencies:**  The bills frequently mandate actions by existing agencies like the California Air Resources Board (CARB), the State Energy Resources Conservation and Development Commission (Energy Commission), and the Ocean Protection Council (OPC).
* **Amending Existing Codes:** Many bills amend existing sections of the Public Utilities Code, the Public Resources Code, and the Vehicle Code, highlighting an incremental approach to policy reform.
* **Setting Deadlines and Targets:**  Bills frequently set specific deadlines for the completion of tasks, the implementation of plans, or the achievement of emission reduction targets.  This provides accountability and structure.


**5. Critical Policy Implications and Potential Impacts:**

* **Economic Impacts:**  Incentives and mandates could influence investment decisions in the energy sector and affect the competitiveness of different energy sources.  Fee extensions might impact vehicle owners.
* **Environmental Impacts:**  The overall impact is expected to be positive in terms of greenhouse gas emissions reduction and improved air quality. However, the effectiveness will depend on the successful implementation of the various plans and mandates.
* **Social Equity:**  The success of programs such as extending energy rate reductions to public housing is crucial for ensuring equitable access to clean and affordable energy.
* **Implementation Challenges:**  The effectiveness of these bills hinges on the resources and capabilities of the involved agencies to implement mandates and roadmaps. Delays or insufficient funding could impede progress.  The complexity of coordinating various agencies across multiple bills presents an implementation challenge.

In conclusion, this cluster of bills reveals a complex and multifaceted approach to energy and utility policy in California.  The success of this policy approach will depend on effective implementation, coordination across agencies, and careful consideration of economic and social equity impacts. The reliance on a mix of mandates, incentives, and regulatory changes suggests a pragmatic approach, but monitoring and evaluation will be essential to determine the overall effectiveness of the legislation.

        """
        
        cluster_2_summary = """
        
Thematic Analysis:
## Analysis of 351 Related Bills on Health Care in California

This analysis examines a cluster of 351 bills focused on health care in California, based on the provided sample of 25 significant bills.  The core thematic elements revolve around access to care, affordability, quality of care, and specific vulnerable populations.  Key policy objectives include expanding coverage, improving affordability, addressing health disparities, and enhancing the quality and safety of healthcare services.

**1. Core Thematic Elements and Policy Objectives:**

The bills address several interconnected themes:

* **Access to Care:** A major theme is expanding access to healthcare services, particularly for underserved populations.  This is evident in bills like AB2200 (CalCare), aiming for universal single-payer healthcare, and SB635, mandating hearing aid coverage for children.  These bills employ policy mechanisms such as creating a new single-payer system (AB2200) and regulatory mandates for insurance coverage (SB635).

* **Affordability:**  Multiple bills target the affordability of healthcare.  SB525 increases minimum wages for healthcare workers, aiming to improve affordability indirectly by improving worker compensation.  SB260 provides a monthly allowance for menstrual products for CalWORKs recipients, directly reducing out-of-pocket costs for a specific vulnerable group.  AB616, the Medical Group Financial Transparency Act, attempts to improve affordability by increasing transparency in physician organization finances, though its direct impact on affordability is less certain.

* **Quality and Safety of Care:**  Several bills address quality and safety.  SB1432 extends deadlines for hospitals to meet seismic safety standards, improving patient safety through infrastructure upgrades. AB2319 mandates implicit bias training for healthcare providers, aiming to reduce racial disparities in maternal healthcare and improve quality of care.  AB3260 streamlines grievance processes for healthcare coverage, improving accountability and access to remedies.

* **Addressing Health Disparities:**  A recurring theme is tackling health disparities.  AB2319 (implicit bias training) directly targets racial disparities in maternal care. SB1016 mandates improved data collection to better understand and address Latino and Indigenous health disparities.  SB282 aims to improve access to care for underserved populations by making changes to Medi-Cal payments for Federally Qualified Health Centers (FQHCs) and Rural Health Clinics (RHCs).

**2. Progression of Policy Approaches:**

The bills show a progression from incremental reforms to more ambitious, systemic changes.  Many bills (e.g., SB635, SB260) focus on specific, targeted improvements within the existing system.  However, AB2200 represents a radical shift towards a universal single-payer system, showcasing a broader, more transformative approach. This suggests a growing movement towards more comprehensive solutions to healthcare challenges.

**3. Key Variations in Approach or Scope:**

The bills vary significantly in their scope and approach:

* **Systemic vs. Targeted Reforms:** AB2200 represents a systemic overhaul, while others focus on specific issues (e.g., hearing aid coverage, seismic safety).
* **Regulatory vs. Financial Mechanisms:**  Some bills use regulatory mandates (e.g., SB635, AB2319), while others employ financial incentives or adjustments (e.g., SB282, SB525).
* **Direct vs. Indirect Impact:** Some bills have a direct impact on healthcare access or affordability (SB260, SB635), while others indirectly influence these factors (SB525, AB616).

**4. Common Regulatory or Implementation Strategies:**

Several common implementation strategies appear:

* **Mandates:** Numerous bills employ mandates to require specific actions by healthcare providers, insurers, or government agencies (e.g., SB635, AB2319, SB1432).
* **Data Collection and Reporting:**  Several bills focus on enhanced data collection and reporting to improve transparency and inform policy decisions (e.g., AB616, SB1016).
* **Amendments to Existing Laws:** Many bills amend existing codes and statutes, reflecting an incremental approach to reform.
* **Establishment of New Boards or Programs:** AB2200 creates a new CalCare Board to govern the single-payer system, illustrating a more transformative strategy involving the creation of new organizational structures.

**5. Critical Policy Implications and Potential Impacts:**

The policy implications are far-reaching:

* **Financial Costs:** AB2200 (CalCare) would have enormous financial implications, requiring significant tax increases or budget reallocation.
* **Administrative Challenges:** Implementing a single-payer system (AB2200) would present significant administrative challenges, including transitioning from a fragmented to a unified system.
* **Potential for Improved Outcomes:**  Bills focusing on quality improvement (AB2319, SB1432) and access expansion (SB635, SB282) could lead to improved health outcomes for specific populations.
* **Political Feasibility:** The ambitious nature of bills like AB2200 raises questions of political feasibility and potential opposition from stakeholders.
* **Unintended Consequences:**  Any significant reform, especially those involving mandates and financial adjustments, could potentially lead to unintended consequences requiring further adjustments.  For example, mandating hearing aid coverage could increase costs for insurers.


In conclusion, this cluster of bills reveals a complex and multifaceted approach to healthcare reform in California.  While some bills pursue incremental improvements within the existing system, others aim for more radical transformations.  Careful consideration of the potential costs, benefits, and unintended consequences of each approach is crucial for effective policymaking.  The interplay between different approaches – systemic vs. targeted, regulatory vs. financial – is a defining feature of this policy landscape.  Further analysis of the full 351 bills would allow for a more detailed understanding of the overall policy direction and its likely impacts.

        """
        
        cluster_3_summary = """
        
Thematic Analysis:
## Analysis of 168 Related Bills: Waste Management, Environmental Protection, and Product Safety in California

This analysis examines a cluster of 168 bills focused on waste reduction, food safety, recycling, pesticide regulation, product safety (including PFAS), and resource management.  The provided sample of 25 bills reveals several core thematic elements and policy approaches.

**1. Core Thematic Elements and Policy Objectives:**

The bills broadly aim to improve environmental sustainability and public health through various policy mechanisms:

* **Waste Reduction & Recycling:** Several bills target single-use plastics (AB2236, SB1053) and beverage containers (SB353), utilizing bans, revised definitions, and expanded recycling programs (Extended Producer Responsibility – EPR – in SB707).  The emphasis is on shifting responsibility for waste management towards producers (EPR) and encouraging composting (ACR161).  AB2236, for example, revises definitions of "single-use carryout bag" to broaden the scope of the existing ban.

* **Pesticide Regulation & Environmental Justice:**  Bills like AB652 and AB1864 address pesticide use, focusing on environmental justice (establishing an advisory committee) and improving notification and reporting requirements near schoolsites to mitigate potential health risks.  AB99 mandates integrated pest management (IPM) for Caltrans, promoting less harmful pest control methods.

* **Product Safety & PFAS Reduction:**  AB246 targets PFAS in menstrual products, employing a ban on sale and manufacture to protect consumer health.  This demonstrates a direct approach to eliminating harmful substances from consumer goods.

* **Climate Change Mitigation:**  SB261 and SB253 address climate-related financial risk and greenhouse gas emission disclosure by large corporations, utilizing mandatory reporting requirements as a mechanism to improve transparency and potentially drive corporate action on climate change.

* **Resource Management & Food Security:**  AB228 focuses on infant formula stockpiles to enhance food security, highlighting a proactive approach to emergency preparedness.

**2. Progression of Policy Approaches Across the Bills:**

The bills demonstrate a progression towards stricter regulations and a greater emphasis on producer responsibility:

* **From Awareness to Regulation:**  ACR161 promotes Compost Awareness Week, while other bills, like AB2236 and SB1053, implement actual bans and stricter regulations on single-use plastics, showing a movement from awareness campaigns to concrete legislative action.

* **Expansion of Recycling Programs:** SB353 expands the California Beverage Container Recycling and Litter Reduction Act, demonstrating a gradual widening of the scope of existing recycling infrastructure.

* **Shifting Producer Responsibility:** SB707's establishment of an EPR program for textiles represents a significant shift in responsibility for waste management from consumers and the state to the producers of those goods.

**3. Key Variations in Approach or Scope:**

The bills vary significantly in their approach and scope:

* **Bans vs. Regulations:** Some bills utilize outright bans (AB246 on PFAS in menstrual products, aspects of AB2236 on single-use bags), while others focus on modifying existing regulations (SB1053, SB353) or introducing new reporting requirements (SB261, SB253, AB1864).

* **Target Industries and Products:** The bills target a wide range of industries and products, from automotive (AB2286, AB316, SB55) to food (AB228, AB2316, AB246), agriculture (AB1864, AB1016), and textiles (SB707).

* **Geographic Scope:** Most bills have statewide application, but some, such as AB1864 (pesticide use near schoolsites), have a more localized focus.

**4. Common Regulatory or Implementation Strategies:**

Several common regulatory and implementation strategies emerge:

* **Mandatory Reporting:**  Several bills rely on mandatory reporting requirements (SB261, SB253, AB1864) to increase transparency and accountability.

* **Establishment of Committees/Agencies:**  AB652 creates an advisory committee, while other bills implicitly involve existing state agencies in implementing the regulations.

* **Amendments to Existing Laws:** Many bills amend existing codes (AB2236, SB1053, AB935), streamlining and updating existing regulatory frameworks.

* **Financial Incentives (implied):** Although not explicitly stated in all samples, several bills (e.g., tax credits in AB52, EPR program in SB707) suggest the potential use of financial incentives or penalties to encourage compliance.

**5. Critical Policy Implications and Potential Impacts:**

* **Economic Impacts:**  Bans on certain products (e.g., AB246, AB2236) may affect manufacturers and retailers, while EPR programs (SB707) may create new costs for producers.  Conversely, some bills may create economic opportunities in the recycling and compost industries.

* **Environmental Impacts:**  The success of these bills hinges on effective implementation.  If successfully implemented, they have the potential to significantly reduce waste, improve air and water quality, and mitigate climate change.

* **Public Health Impacts:**  Regulations on pesticides and PFAS (AB652, AB1864, AB246) will directly improve public health by reducing exposure to harmful chemicals.  Food safety regulations (AB2316) will similarly protect children’s health.

* **Equity and Justice:**  The focus on environmental justice in AB652 and the consideration of community impacts in some bills aim to ensure that the benefits and burdens of these policies are distributed equitably.

**Conclusion:**

The 168 bills collectively represent a multifaceted approach to improving environmental sustainability, public health, and resource management in California.  The bills progress from awareness campaigns to stringent regulations, reflecting a growing commitment to tackling environmental challenges and protecting public health through diverse regulatory mechanisms and a growing emphasis on producer responsibility.  However, the success of these ambitious policies depends on effective implementation and careful consideration of their economic and social impacts.  Further analysis of the full 168 bills would provide a more comprehensive understanding of the specific nuances and potential synergies between these individual pieces of legislation.

        """
        
        cluster_4_summary = """
        
Thematic Analysis:
## Analysis of 351 Related Bills: California Legislative Cluster

This analysis examines a cluster of 351 bills focusing on civil, information, privacy, consumer, platform, media, personal, AI, social media, and digital issues.  Based on the provided sample of 25 significant bills, several core thematic elements and policy objectives emerge.

**1. Core Thematic Elements and Policy Objectives:**

The bills reveal a multifaceted approach to regulating emerging technologies and protecting various societal interests. Key themes include:

* **Worker Protection:** A significant portion of the bills, exemplified by AB2288, SB92, and SCA7, focus on strengthening worker rights and protections.  These bills utilize policy mechanisms such as amending the Labor Code (AB2288, SB92) to modify the Private Attorneys General Act (PAGA) – impacting employee lawsuits and employer responsibilities – and proposing constitutional amendments to guarantee unionization rights (SCA7).

* **Child Safety & Online Harms:** Several bills (AB1831, SB1381, AB2839) directly address the harms associated with AI-generated content, particularly child sexual abuse material (CSAM) and deceptive political advertising.  The policy mechanism here is expanding penal codes to criminalize the creation, distribution, and possession of AI-generated CSAM and prohibiting the knowing distribution of AI-generated deceptive political advertisements.

* **Platform Regulation:**  Bills like SB1144 aim to regulate online marketplaces, revising definitions of key terms like "high-volume third-party seller" to enhance oversight and potentially increase accountability for online platforms. The policy mechanism is statutory amendment to adjust the definition and scope of existing regulations.

* **Consumer Protection:**  Bills like AB473 aim to protect consumers from anti-competitive practices by motor vehicle manufacturers, distributors, and dealers. The policy mechanism involves prohibiting specific actions like vehicle allocation to avoid dealer conflict.

* **Data Privacy & Security:** While not explicitly detailed in the sample, the presence of themes like "information privacy" and "digital equity" suggests that the full cluster likely addresses data privacy concerns. The potential policy mechanisms could include data breach notification requirements or data minimization strategies.


**2. Progression of Policy Approaches Across the Bills:**

The bills show a progression towards proactive regulation of emerging technologies.  Earlier bills (e.g., those related to PAGA) primarily focus on refining existing legal frameworks. Later bills (e.g., those addressing AI-generated CSAM) directly address harms arising from new technologies, showing a shift from reactive to preventative legislative strategies.  This progression signifies a developing understanding of the societal implications of new technologies and a need for tailored legal frameworks.


**3. Key Variations in Approach or Scope:**

The bills vary significantly in their scope and approach:

* **Specificity vs. Broad Principles:** Some bills, such as those related to PAGA, are highly specific in their amendments to existing laws. Others, like the Digital Equity Bill of Rights (AB414), establish broad principles that need further legislative action for concrete implementation.

* **Civil vs. Criminal Penalties:**  Some bills focus on civil remedies (e.g., AB452, removing statute of limitations for childhood sexual assault lawsuits), while others focus on criminal penalties (e.g., AB1831, criminalizing AI-generated CSAM).  This reflects the diverse nature of the harms addressed.

* **Targeted vs. Broad Regulation:** SB1144 targets online marketplaces, showcasing targeted regulation, while AB414 adopts a broader approach by aiming for digital equity across the board.

**4. Common Regulatory or Implementation Strategies:**

Several common regulatory strategies are observed:

* **Statutory Amendments:** Many bills utilize the approach of amending existing statutes (Labor Code, Penal Code, Vehicle Code, etc.) to incorporate new provisions or modify existing ones.

* **Definition Revision:** Some bills focus on clarifying or revising the definition of key terms (e.g., "online marketplace," "high-volume third-party seller") to ensure effective application of existing laws in new contexts.

* **Agency Oversight:**  The creation of the "Unflavored Tobacco List" (AB3218) implies potential agency oversight and enforcement mechanisms.

**5. Critical Policy Implications and Potential Impacts:**

The bills’ implementation will have significant impacts:

* **Increased Litigation:** Amendments to PAGA (AB2288, SB92) could potentially lead to an increase in wage and hour lawsuits.

* **Technological Innovation:**  Regulation of AI-generated content (AB1831, SB1381) could impact the development and deployment of AI technologies, requiring developers to consider legal compliance.

* **Platform Accountability:**  Regulation of online marketplaces (SB1144) could increase platform accountability for the activities of third-party sellers.

* **Economic Impacts:**  Bills affecting labor rights (SCA7) and businesses (AB473, SB1144) could have significant economic implications for employers and consumers.

* **Civil Rights & Liberties:** Bills focused on civil rights (AB3024) and digital equity (AB414) could have far-reaching implications for protecting fundamental rights and reducing the digital divide.

The full cluster's impact will depend on the specifics of the 351 bills and their interaction.  A comprehensive assessment requires a detailed examination of all bills, not just the sample provided.  Furthermore, unintended consequences and the effectiveness of implementation mechanisms need further consideration.


        """
        
        cluster_5_summary = """
        
Thematic Analysis:
## Analysis of 180 Related Bills on Housing in California

This analysis examines a cluster of 180 bills focused on housing in California, based on the provided sample of 25 significant bills.  The core themes – housing, units, rent, landlords, homelessness, rental, residential, tenants, affordable, and affordable housing – reveal a multifaceted approach to addressing a significant state-wide crisis.

**1. Core Thematic Elements and Policy Objectives:**

The bills primarily target three interconnected areas: **increasing the supply of affordable housing**, **enhancing tenant protections**, and **combating homelessness**.  Policy mechanisms employed vary significantly:

* **Increasing Affordable Housing Supply:** This is the dominant theme.  Several bills utilize **financial mechanisms**, such as AB 1657's proposed $10 billion bond to fund affordable housing initiatives, and AB 1319's modification of the Bay Area Housing Finance Authority to potentially increase housing revenue.  Others focus on **streamlining development**, for example, AB 1332's mandate for pre-approved ADU plans and AB 2729's extension of housing entitlements. AB 309 proposes a novel approach with the creation of a Social Housing Program.

* **Enhancing Tenant Protections:** Bills like SB 567 address **eviction protections** by modifying "just cause" requirements and regulating rent increases.  AB 59 aims to increase **financial relief** for renters via a larger renter's tax credit.  SB 644 and SB 683 focus on **transparency and consumer protection** in short-term rentals and hotel bookings by mandating clear pricing and cancellation policies.

* **Combating Homelessness:**  This theme is addressed through direct funding (AB 799's funding for the Interagency Council on Homelessness), pilot programs (SB 37's pilot program for older adults and adults with disabilities), and regulatory adjustments like AB 42's temporary suspension of fire sprinkler requirements for temporary homeless shelters.


**2. Progression of Policy Approaches Across the Bills:**

The bills demonstrate a progression from focusing on isolated problems to a more holistic, multi-pronged approach.  Earlier bills might address specific issues (e.g.,  renter's tax credit in AB 59,  short-term rental regulations in AB 537).  Later bills, especially AB 1657 and AB 309, signal a shift toward large-scale, systemic change through significant funding and the creation of new programs.  This suggests a growing recognition that piecemeal solutions are insufficient to tackle the housing crisis.

**3. Key Variations in Approach or Scope:**

Significant variations exist in the scope and approach:

* **Geographic Scope:** Some bills apply statewide (e.g., AB 1657, SB 567), while others focus on specific regions (e.g., AB 1319 affecting the Bay Area).
* **Target Population:**  While many bills broadly target affordability, some specifically address the needs of particular groups like older adults and those with disabilities (SB 37) or renters (AB 59).
* **Policy Instrument:**  The range of policy instruments is broad, including financial incentives, regulatory changes, mandates, pilot programs, and even constitutional amendments (ACA 10).


**4. Common Regulatory or Implementation Strategies:**

Several common strategies emerge:

* **Mandates:** Many bills employ mandates to require specific actions from local agencies (e.g., AB 1332 on ADU pre-approval, AB 653 on public housing authority reporting).
* **Data Collection and Reporting:**  Several bills emphasize increased data collection and reporting to improve transparency and inform policy decisions (AB 653, AB 1820).
* **Amendments to Existing Codes:**  A significant number of bills amend existing codes in the Government Code, Business and Professions Code, and Revenue and Taxation Code, indicating an incremental approach to policy change.


**5. Critical Policy Implications and Potential Impacts:**

The implications are significant and multifaceted:

* **Fiscal Impacts:**  AB 1657's $10 billion bond will require substantial taxpayer investment. The success of this approach hinges on effective program design and implementation to prevent wasteful spending.
* **Housing Affordability:**  While the bills aim to improve affordability, the actual impact will depend on factors like construction costs, market dynamics, and effective implementation of programs.
* **Regulatory Burden:**  The numerous mandates placed on local agencies could create a significant regulatory burden, potentially slowing down the development process.
* **Equity Concerns:**  While many bills aim to improve equity, careful attention must be paid to ensuring that programs effectively reach vulnerable populations and don't inadvertently exacerbate existing inequalities.
* **Effectiveness of New Programs:**  The success of novel programs like the Social Housing Program (AB 309) hinges on effective design, adequate funding, and efficient management.

In conclusion, the cluster of bills reveals a comprehensive, though complex, strategy to address California's housing crisis.  Success will depend on careful implementation, sufficient funding, effective collaboration among stakeholders, and ongoing monitoring of the programs' impacts to ensure they deliver on their promises.  The mix of large-scale funding initiatives, targeted regulations, and incremental code amendments reflects a multifaceted approach to tackle a problem of significant scope and complexity.  Further research is needed to assess the effectiveness of these diverse policy interventions.

        """
        
        # Cluster summaries
        cluster_info = {
            0: {"themes": "water, projects, tax, budget, wildfire, groundwater, board, flood, drought, government",
                "size": 228,
                "summary": cluster_0_summary,
                "party_dist": "D: 183, R: 25, Unknown: 20"},
            1: {"themes": "energy, utilities, commission, public utilities, puc, electrical, electric, public, utilities commission, corporations",
                "size": 159,
                "summary": cluster_1_summary,
                "party_dist": "D: 142, R: 14, Unknown: 3"},
            2: {"themes": "health, care, health care, services, medical, coverage, treatment, plans, medi, medi cal",
                "size": 351,
                "summary": cluster_2_summary,
                "party_dist": "D: 307, R: 29, Unknown: 15"},
            3: {"themes": "waste, food, recycling, pesticide, products, pfas, plastic, containers, recovery, resources",
                "size": 168,
                "summary": cluster_3_summary,
                "party_dist": "D: 152, R: 11, Unknown: 5"},
            4: {"themes": "civil, information, privacy, consumer, platforms, media, personal, ai, social media, digital",
                "size": 351,
                "summary": cluster_4_summary,
                "party_dist": "D: 295, R: 51, Unknown: 5"},
            5: {"themes": "housing, units, rent, landlords, homelessness, rental, residential, tenants, affordable, affordable housing",
                "size": 180,
                "summary": cluster_5_summary,
                "party_dist": "D: 158, R: 21, Unknown: 1"}
        }
        
        # Create an expander for each cluster
        for cluster_id in sorted(cluster_info.keys()):
            with st.expander(f"Cluster {cluster_id}: {cluster_map[cluster_id]}", expanded=True):
                info = cluster_info[cluster_id]
                st.markdown(f"""
                **Size:** {info['size']} bills  
                **Key Themes:** {info['themes']}  
                **Party Distribution:** {info['party_dist']}
                **Summary:** {info['summary']}
                """)
        
        # Add cluster statistics visualization
        st.subheader("Cluster Sizes")
        sizes_df = pd.DataFrame([
            {'Cluster': f"{i}: {cluster_map[i]}", 'Size': info['size']}
            for i, info in cluster_info.items()
        ])
        
        fig_sizes = px.bar(
            sizes_df,
            x='Cluster',
            y='Size',
            title='Bills per Cluster'
        )
        
        fig_sizes.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig_sizes, use_container_width=True)

        # Add party distribution visualization
        st.subheader("Party Distribution by Cluster")
        party_data = []
        for cluster_id, info in cluster_info.items():
            dist = info['party_dist'].split(', ')
            for party_count in dist:
                party, count = party_count.split(': ')
                party_data.append({
                    'Cluster': f"{cluster_id}: {cluster_map[cluster_id]}",
                    'Party': party,
                    'Count': int(count)
                })
                
        party_df = pd.DataFrame(party_data)
        
        fig_party = px.bar(
            party_df,
            x='Cluster',
            y='Count',
            color='Party',
            title='Party Distribution across Clusters',
            barmode='stack'
        )
        
        fig_party.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig_party, use_container_width=True)
    
def show_company_details(company_data, bills_lookup, selected_company):
    """Display the company-specific details"""
    data = company_data[selected_company]
    
    st.header(f"{selected_company} - Company Analysis")
    
    # Create two columns for the top metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Lobbying Spending", f"${data['total_spending']:,.2f}")
        
    with col2:
        st.metric("Number of Bills Lobbied", len(data['bills']))
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Top Topics", "Bills", "Spending Over Time"])
    
    with tab1:
        # Create bar chart of top topics
        if data['all_tags']:
            tag_counts = Counter(data['all_tags'])
            top_tags = dict(tag_counts.most_common(15))
            
            # Create a DataFrame from the tag counts
            df_tags = pd.DataFrame({
                'Topic': list(top_tags.keys()),
                'Count': list(top_tags.values())
            })
            
            fig = px.bar(
                df_tags,
                x='Topic',
                y='Count',
                title="Top 15 Lobbying Topics"
            )
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available for this company")
        
    with tab2:
        # Show bills with summaries and tags
        st.subheader("Bills and Their Summaries")
        
        for bill in sorted(data['bills']):
            clean_bill = clean_bill_number(bill)
            if clean_bill in bills_lookup:
                with st.expander(f"{bill}"):
                    st.write(bills_lookup[clean_bill]['gemini_summary'])
                    st.write("**Tags:**")
                    if isinstance(bills_lookup[clean_bill]['tags'], list):
                        for tag in bills_lookup[clean_bill]['tags']:
                            st.caption(f"#{tag}")
                    st.divider()
                    
    with tab3:
        # Create spending over time visualization
        if data['quarterly_spending']:
            spending_df = pd.DataFrame(data['quarterly_spending'])
            spending_df = spending_df.sort_values('date')
            
            fig = px.line(
                spending_df,
                x='date',
                y='amount',
                title="Quarterly Lobbying Spending",
                labels={'date': 'Quarter Start', 'amount': 'Amount ($)'}
            )
            fig.update_layout(yaxis_tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-series spending data available")

def main():
    # Load data
    with st.spinner('Loading data...'):
        company_data, bills_lookup, bill_data, unique_companies, bills_df = load_data()
    
    # Create tabs for main navigation
    intro_tab, cluster_tab, company_tab = st.tabs([
        "Introduction & Analysis",
        "Cluster Visualization",
        "Company Analysis"
    ])
    
    with intro_tab:
        show_introduction(company_data)
        show_bill_overview(bill_data)
    
    with cluster_tab:
        show_cluster_visualization(bills_df)
    
    with company_tab:
        # Company selector using the unique companies from the data
        all_companies = sorted(company_data.keys())
        selected_company = st.selectbox(
            "Select Company",
            all_companies,
            index=0
        )
        
        if selected_company:
            show_company_details(company_data, bills_lookup, selected_company)

if __name__ == "__main__":
    main()