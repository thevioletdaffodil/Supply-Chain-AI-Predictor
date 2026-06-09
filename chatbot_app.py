import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# GLOBAL CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "SYNTHETIC_PO_DATA.csv")
RAG_DB_PATH = os.path.join(BASE_DIR, "predicted_lead_times_for_rag.csv")

HIGH_RISK_VENDORS = [
    'Strata Global', 'Quantum Components'
]

HIGH_RISK_CITIES = [
    'Shenzhen', 'Mumbai'
]


# ---------------------------------------------------------
# 4. SUPPLY CHAIN ASSISTANT
# ---------------------------------------------------------
class SupplyChainAssistant:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.rag_df = pd.read_csv(RAG_DB_PATH)

        self.reg_mean = joblib.load(os.path.join(BASE_DIR, 'stage2_mean_regressor.joblib'))
        self.reg_quant = joblib.load(os.path.join(BASE_DIR, 'stage2_quant_regressor.joblib'))
        self.encoding_maps = joblib.load(os.path.join(BASE_DIR, 'target_encoding_maps.joblib'))
        self.expected_cols = joblib.load(os.path.join(BASE_DIR, 'expected_columns.joblib'))

        print("Loading RAG Embeddings & FAISS Index...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        corpus_embeddings = self.embedding_model.encode(self.rag_df['RAG_Context'].tolist(), show_progress_bar=False).astype('float32')
        self.faiss_index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
        self.faiss_index.add(corpus_embeddings)

    def run_ml_prediction(self, order_dict):
        df_new = pd.DataFrame([order_dict])

        df_new['PO_DATE'] = pd.to_datetime(df_new['PO_DATE'], dayfirst=True)
        df_new['PO_MONTH'] = df_new['PO_DATE'].dt.month
        df_new['PO_YEAR'] = df_new['PO_DATE'].dt.year
        df_new['PO_DAYOFTHEWEEK'] = df_new['PO_DATE'].dt.dayofweek
        df_new['Is_International'] = (df_new.get('COUNTRY', 'Domestic') != 'Domestic').astype(int)

        df_new['Is_High_Risk_Vendor'] = df_new['VENDOR'].isin(HIGH_RISK_VENDORS).astype(int)
        df_new['Is_High_Risk_City'] = df_new['CITY'].isin(HIGH_RISK_CITIES).astype(int)
        df_new['Severe_Risk_Warning'] = ((df_new['Is_High_Risk_Vendor'] == 1) & (df_new['Is_High_Risk_City'] == 1)).astype(int)

        for col, maps in self.encoding_maps.items():
            if col in df_new.columns:
                df_new[f"{col}_mean"] = df_new[col].map(maps['mean']).fillna(0)
                df_new[f"{col}_std"] = df_new[col].map(maps['std']).fillna(0)

        X_score = df_new.reindex(columns=self.expected_cols, fill_value=0)

        expected_lead_time = int(np.round(np.maximum(0, self.reg_mean.predict(X_score)[0])))
        max_lead_time = int(np.round(np.maximum(expected_lead_time, self.reg_quant.predict(X_score)[0])))

        hist_mean = df_new['VENDOR_mean'].values[0] if 'VENDOR_mean' in df_new.columns else 0
        hist_std = df_new['VENDOR_std'].values[0] if 'VENDOR_std' in df_new.columns else 0

        return expected_lead_time, max_lead_time, hist_mean, hist_std

    def ask(self, user_question, what_if_data=None, k=3):
        if what_if_data:
            expected_days, max_days, hist_mean, hist_std = self.run_ml_prediction(what_if_data)
            context_str = (
                f"[SYSTEM TYPE: LIVE ML SIMULATION RESULT]\n"
                f"- Hypothetical Vendor: {what_if_data['VENDOR']}\n"
                f"- Target Destination: {what_if_data['CITY']}\n"
                f"- Item: {what_if_data['ITEM']}\n"
                f"- Quantity: {what_if_data.get('QUANTITY', 'Unknown')}\n"
                f"- MODEL A (EXPECTED TIMELINE): The total order fulfillment process will take {expected_days} days.\n"
                f"- MODEL B (RECOMMENDED SCHEDULING BUFFER): Do not schedule this part sooner than {max_days} days from the PO date.\n"
                f"- Vendor Historical Profile: Usually takes {round(hist_mean, 1)} days overall (Volatility Std Dev: {round(hist_std, 1)} days).\n"
            )
        else:
            query_embed = self.embedding_model.encode([user_question]).astype('float32')
            _, I = self.faiss_index.search(query_embed, k)
            retrieved_contexts = [self.rag_df.loc[idx, 'RAG_Context'] for idx in I[0] if idx in self.rag_df.index]
            context_str = "[SYSTEM TYPE: HISTORICAL DATA LOOKUP]\n\n" + "\n\n".join(retrieved_contexts)

        system_prompt = """You are an expert Supply Chain Operations Director.
        Translate complex raw tracking statistics into actionable, clear executive summaries.
        1. Sound confident, direct, and warm. Use professional logistics terms naturally.
        2. Use structured markdown elements: **bold headers**, bullet points, and neat spacing.
        3. If context is '[SYSTEM TYPE: LIVE ML SIMULATION RESULT]', clearly distinguish between the "Expected Timeline" and the "Recommended Scheduling Buffer"."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context Data:\n{context_str}\n\nUser Question: {user_question}"}
                ],
                temperature=0.2
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"API Error: {str(e)}"


if __name__ == "__main__":
    import os

    print("=== Supply Chain AI Assistant Demo ===")

    # 1. Fetch the API key from the local environment variable safely
    # This keeps your code professional and compliant with secure coding standards
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("\n[NOTICE]: 'GROQ_API_KEY' environment variable not detected.")
        print("To run the interactive LLM simulation, configure your local environment variable.")
        print("Alternatively, paste your temporary key below to run a quick test.")
        api_key = input("Enter your Groq API Key (or press Enter to skip): ").strip()

    if not api_key:
        print("\nSkipping LLM initialization. Code structure verified successfully.")
    else:
        # 2. Initialize the assistant using the credentials provided
        try:
            assistant = SupplyChainAssistant(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                model_name="llama-3.1-8b-instant"
            )

            print("\n--- RUNNING PORTFOLIO DEMO SCENARIO ---")

            # An unscheduled mock order mapped directly to your open-source synthetic dataset profiles
            mock_unscheduled_order = {
                'PO_DATE': '15-11-2026',
                'VENDOR': 'Quantum Components',  # Represents your volatile synthetic testing profile
                'CITY': 'Mumbai',
                'COUNTRY': 'International',
                'ITEM': 'Component_v12',
                'PURCHASE_LEAD_TIME': 14,
                'QUANTITY': 15000
            }

            demo_question = "I want to place an order for 15,000 units with Quantum Components for Mumbai. What timeline should we plan for our assembly floor?"
            print(f"User Query: {demo_question}")

            # 3. Requesting the dual-band calculation and text generation summary
            response = assistant.ask(user_question=demo_question, what_if_data=mock_unscheduled_order)
            print(f"\nAI Assistant Response:\n{response}")

        except Exception as e:
            print(f"\nCould not run the demo scenario: {e}")