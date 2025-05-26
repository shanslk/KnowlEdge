from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import knowledge_rules

app = Flask(__name__)

# Define the uploads folder path (make sure it is in your project root)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
CRM_CSV = os.path.join(UPLOAD_FOLDER, 'crm_data_cleaned_agent_view.csv')
CSAT_CSV = os.path.join(UPLOAD_FOLDER, 'csat_data_randomized_comments.csv')
KB_CSV = os.path.join(UPLOAD_FOLDER, 'dell_enhanced_kb_articles_500.csv')

# Load CSV files using pandas
crm_df = pd.read_csv(CRM_CSV)
csat_df = pd.read_csv(CSAT_CSV)
kb_df = pd.read_csv(KB_CSV)

# Pass the dataframes to the knowledge rules module
knowledge_rules.load_data(kb_df, crm_df, csat_df)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        query = data.get("message", "")
        # Process query to search for KB articles and generate a response
        response_text = knowledge_rules.process_query(query)
        return jsonify({"response": response_text})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)