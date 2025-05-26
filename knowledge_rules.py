import pandas as pd
import re
import numpy as np

# Import scikit-learn models for machine learning ranking.
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# Global data variables to hold preloaded CSV data.
kb_articles = None
crm_data = None
csat_data = None

# Parameters for composite scoring.
CSAT_WEIGHT = 10
RESOLVED_WEIGHT = 1
GOLDEN_PENALTY = 1e6  # Penalty applied if KB article was previously tried and failed for this customer.

def load_data(kb_df: pd.DataFrame, crm_df: pd.DataFrame, csat_df: pd.DataFrame) -> None:
    """
    Loads the preloaded CSV data into module-level variables.
    
    Parameters:
      kb_df (pd.DataFrame): DataFrame for KB articles.
      crm_df (pd.DataFrame): DataFrame for CRM data.
      csat_df (pd.DataFrame): DataFrame for CSAT feedback.
    """
    global kb_articles, crm_data, csat_data
    kb_articles = kb_df
    crm_data = crm_df
    csat_data = csat_df

def retrieve_kb_articles(issue_type_query: str) -> list:
    """
    Filters KB articles based on the provided issue type.
    
    The CSV file (dell_enhanced_kb_articles_500.csv) is expected to include columns:
        kb_id, title, issue_type, content, tags, created_date, updated_date, search_keywords

    Parameters:
      issue_type_query (str): The issue type to search for (e.g., "Battery Not Charging").
      
    Returns:
      list: A list of matching articles represented as dictionaries.
    """
    global kb_articles
    if kb_articles is None:
        return []
    
    mask = kb_articles['issue_type'].str.contains(issue_type_query, case=False, na=False)
    filtered = kb_articles[mask]
    return filtered.to_dict(orient="records")

def extract_customer_id(query: str) -> str:
    """
    Extracts a customer ID from the query if provided.
    Example: "Customer ID: Shan1234" returns "Shan1234".
    
    Parameters:
      query (str): The user's query.
      
    Returns:
      str: The extracted customer ID or an empty string if not found.
    """
    pattern = r"customer id\s*:\s*([A-Za-z0-9]+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def compute_composite_score(avg_csat: float, resolved_count: int, prior_failed: bool) -> float:
    """
    Computes the composite score using the provided parameters.
    CSAT is only incorporated if avg_csat is at least 4.
    
    Parameters:
      avg_csat (float): Average CSAT score.
      resolved_count (int): Count of tickets resolved.
      prior_failed (bool): Flag if the KB article was previously provided and failed for this customer.
      
    Returns:
      float: The composite score.
    """
    csat_component = CSAT_WEIGHT * avg_csat if avg_csat >= 4 else 0
    penalty = GOLDEN_PENALTY if prior_failed else 0
    return csat_component + (RESOLVED_WEIGHT * resolved_count) - penalty

def extract_features_for_candidate(candidate: dict, customer_id: str) -> dict:
    """
    For a given candidate KB article, compute the features:
      - avg_csat (average CSAT score for resolved tickets)
      - resolved_count (number of resolved tickets for this article)
      - failed_flag (1 if previously tried and failed for this customer, else 0)
    
    Also computes the composite score as the target value.
    
    Returns:
      dict: A dictionary containing features and the composite target.
    """
    kb_id = candidate.get("kb_id")
    
    # Default feature values.
    avg_csat = 0
    resolved_count = 0
    explanation = []

    # CSAT data: Filter for records matching kb_id and with resolution_status "resolved".
    if csat_data is not None:
        csat_subset = csat_data[csat_data['kb_article_id'] == kb_id]
        csat_resolved = csat_subset[csat_subset['resolution_status'].str.lower() == "resolved"]
        if not csat_resolved.empty:
            avg_csat = csat_resolved['csat_score'].mean()
            resolved_count = len(csat_resolved)
            explanation.append(f"Avg CSAT: {avg_csat:.2f} from {resolved_count} resolved tickets")
        else:
            explanation.append("No CSAT data for resolved tickets")
    else:
        explanation.append("CSAT data unavailable")
    
    # CRM data: Check if article was previously provided and failed for this customer.
    prior_failed = False
    if customer_id and crm_data is not None:
        crm_subset = crm_data[(crm_data['ticket_id'].notna()) & 
                              (crm_data['customer_id'].str.lower() == customer_id.lower())]
        if not crm_subset.empty:
            for _, row in crm_subset.iterrows():
                resolution = str(row.get('resolution_type', ""))
                status = str(row.get('resolution_status', "")).lower()
                if kb_id in resolution and status != "resolved":
                    prior_failed = True
                    explanation.append("Previously tried and failed for this customer")
                    break

    # Compute composite score (target value).
    composite_score = compute_composite_score(avg_csat, resolved_count, prior_failed)
    
    return {
        "avg_csat": avg_csat,
        "resolved_count": resolved_count,
        "failed_flag": int(prior_failed),
        "composite_score": composite_score,
        "explanation": "; ".join(explanation)
    }

def select_and_train_model(candidates_features: list):
    """
    Given a list of candidate features (each is a dict with avg_csat, resolved_count, failed_flag, and composite_score),
    decide whether the relationship is linear. If linear (absolute Pearson correlation above threshold for each feature)
    use Linear Regression; otherwise use Random Forest.
    
    Returns:
      model: The trained ML model.
      model_type: A string representing the chosen model type.
    """
    # Prepare the design matrix X and target vector y.
    X = []
    y = []
    for feat in candidates_features:
        X.append([feat["avg_csat"], feat["resolved_count"], feat["failed_flag"]])
        y.append(feat["composite_score"])
    X = np.array(X)
    y = np.array(y)
    
    # Check correlation per feature.
    correlations = []
    for col in range(X.shape[1]):
        if np.std(X[:, col]) > 0:
            corr, _ = pearsonr(X[:, col], y)
            correlations.append(abs(corr))
        else:
            correlations.append(0)
    # Define threshold for linearity.
    threshold = 0.5
    if all(corr >= threshold for corr in correlations):
        model = LinearRegression()
        model_type = "Linear Regression"
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model_type = "Random Forest"
    model.fit(X, y)
    return model, model_type

def rank_kb_articles(candidates: list, customer_id: str) -> list:
    """
    Ranks candidate KB articles using a machine learning model.
    
    For each candidate, the function computes features (avg_csat, resolved_count, failed_flag)
    and a composite score as ground truth. It then trains either a multivariable regression model
    or a Random Forest model based on the linearity of the data, and uses the model to predict
    scores for ranking.
    
    Each candidate is augmented with:
      - 'ml_score': The score predicted by the model.
      - 'ranking_explanation': Explanation detailing feature values and the model used.
    
    Parameters:
      candidates (list): A list of candidate KB articles (dictionaries).
      customer_id (str): The customer ID provided by the agent (if any).
    
    Returns:
      list: The candidate KB articles sorted in descending order by ml_score.
    """
    candidates_features = []
    for candidate in candidates:
        features = extract_features_for_candidate(candidate, customer_id)
        candidate["extracted_features"] = features
        candidates_features.append(features)
    
    # Train our model based on feature-target data.
    model, model_type = select_and_train_model(candidates_features)
    
    # Predict score for each candidate.
    X_pred = []
    for feat in candidates_features:
        X_pred.append([feat["avg_csat"], feat["resolved_count"], feat["failed_flag"]])
    X_pred = np.array(X_pred)
    predicted_scores = model.predict(X_pred)
    
    ranked_candidates = []
    for candidate, score, feat in zip(candidates, predicted_scores, candidates_features):
        candidate['ml_score'] = score
        candidate['ranking_explanation'] = (
            f"{feat['explanation']}. Model used: {model_type}."
        )
        ranked_candidates.append(candidate)
    
    ranked_candidates.sort(key=lambda art: art['ml_score'], reverse=True)
    return ranked_candidates

def process_query(query: str) -> str:
    """
    Processes the agent's query.
    
    The function extracts the issue type and checks for an optional customer ID from the query.
    If the customer ID is not provided, it returns a prompt asking for the customer ID.
    Once both are provided, it retrieves candidate KB articles matching the issue type,
    ranks them using the ML-based ranking matrix, and returns details about the top recommended KB article.
    
    Parameters:
      query (str): The agent's query.
      
    Returns:
      str: A response message with the top-ranked KB article details and ranking explanation,
           or an error/instruction message if no issue type or customer ID is recognized.
    """
    # List of known issue types extracted from our KB CSV.
    possible_issue_types = [
        "Battery Not Charging", "Cannot Connect to Internet", "Slow Performance",
        "No Sound", "Touchpad Not Responding", "No POST", "Overheating",
        "Blue Screen Error", "Keyboard Not Working", "Display Flickering"
    ]
    
    matched_issue = None
    for issue in possible_issue_types:
        if issue.lower() in query.lower():
            matched_issue = issue
            break

    if not matched_issue:
        return ("Please specify an issue type in your query. " +
                "Examples: " + ", ".join(possible_issue_types))
    
    # Extract customer ID if provided.
    customer_id = extract_customer_id(query)
    if not customer_id:
        # If the customer ID is missing, prompt the agent for it.
        return "Please provide the customer ID."
    
    # Retrieve candidate KB articles.
    candidates = retrieve_kb_articles(matched_issue)
    if not candidates:
        return f"Sorry, no KB articles found for the issue type: {matched_issue}"
    
    # Rank candidates using the ML-based ranking system.
    ranked_candidates = rank_kb_articles(candidates, customer_id)
    top_article = ranked_candidates[0]
    title = top_article.get("title", "No Title")
    content = top_article.get("content", "")
    explanation = top_article.get("ranking_explanation", "")
    
    response = (
        f"KB Article Recommendation for '{matched_issue}':\n"
        f"Title: {title}\n"
        f"Snippet: {content[:200]}...\n\n"
        f"Ranking Explanation: {explanation}"
    )
    response += f"\n(Note: Results tailored for customer ID {customer_id})."
    
    return response

def get_customer_feedback(ticket_id: str) -> list:
    """
    Retrieves customer feedback for a given ticket from the CSAT data.
    
    Parameters:
      ticket_id (str): The ticket identifier.
      
    Returns:
      list: A list of customer feedback comments associated with the given ticket.
    """
    global csat_data
    if csat_data is None:
        return []
    feedback_df = csat_data[csat_data['ticket_id'] == ticket_id]
    return feedback_df['comments'].tolist()

def get_crm_ticket_details(ticket_id: str) -> dict:
    """
    Retrieves CRM ticket details for a given ticket from the CRM data.
    
    Parameters:
      ticket_id (str): The ticket identifier.
      
    Returns:
      dict: A dictionary of ticket details (e.g., customer name, issue type, etc.).
    """
    global crm_data
    if crm_data is None:
        return {}
    ticket_df = crm_data[crm_data['ticket_id'] == ticket_id]
    if ticket_df.empty:
        return {}
    return ticket_df.iloc[0].to_dict()