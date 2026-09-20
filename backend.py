import io
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FinPilot Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# IN-MEMORY DATABASE
# -------------------------------------------------------------------
TRANSACTIONS_DB: List[Dict] = []
GOALS_DB: List[Dict] = [
    {"id": 1, "name": "Emergency Fund", "target": 10000.0, "current": 4500.0},
    {"id": 2, "name": "Vacation", "target": 3000.0, "current": 1200.0},
]

# -------------------------------------------------------------------
# CATEGORIZATION ENGINE & RULES
# -------------------------------------------------------------------
CATEGORY_RULES = {
    "Housing": ["rent", "mortgage", "hoa", "lease"],
    "Groceries": ["walmart", "whole foods", "trader joe", "safeway", "kroger", "supermarket", "grocery"],
    "Dining Out": ["starbucks", "mcdonalds", "chipotle", "uber eats", "restaurant", "cafe", "food"],
    "Subscriptions": ["netflix", "spotify", "hulu", "apple.com", "amazon prime", "chatgpt"],
    "Utilities": ["electric", "water", "gas", "comcast", "verizon", "att"],
    "Transport": ["uber", "lyft", "shell", "chevron", "gas station", "subway"],
    "Income": ["payroll", "salary", "direct deposit", "stripe", "employer", "deposit", "refund"],
}

# Category synonym map for natural query understanding
SYNONYMS = {
    "food": ["Groceries", "Dining Out"],
    "groceries": ["Groceries"],
    "dining": ["Dining Out"],
    "eating out": ["Dining Out"],
    "travel": ["Transport"],
    "transportation": ["Transport"],
    "rent": ["Housing"],
    "bills": ["Utilities"],
}

def is_income_transaction(description: str, original_amount: float) -> bool:
    desc_str = str(description or "").strip().lower()
    if original_amount < 0:
        return False
    income_keywords = CATEGORY_RULES["Income"]
    return any(keyword in desc_str for keyword in income_keywords)

def categorize_transaction(description: str, amount: float) -> str:
    desc_str = str(description or "").strip().lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in desc_str for keyword in keywords):
            return category
    return "Income" if amount > 0 else "Uncategorized"

# -------------------------------------------------------------------
# DATA INGESTION & PROCESSING
# -------------------------------------------------------------------
def process_csv_dataframe(df: pd.DataFrame) -> List[Dict]:
    df.columns = [str(col).strip().lower() for col in df.columns]

    if "amount" not in df.columns and "transaction amount" not in df.columns:
        if "debit" in df.columns and "credit" in df.columns:
            for col in ["debit", "credit"]:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[^\d.-]", "", regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df["amount"] = df["credit"] - df["debit"].abs()

    col_map = {}
    for col in df.columns:
        if any(k in col for k in ["date", "time", "posted", "txn_date"]):
            col_map[col] = "date"
        elif any(k in col for k in ["desc", "memo", "payee", "name", "detail", "narration"]):
            col_map[col] = "description"
        elif any(k in col for k in ["amount", "value", "sum", "total"]):
            col_map[col] = "amount"

    df = df.rename(columns=col_map)
    required = ["date", "description", "amount"]
    
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Detected columns: {list(df.columns)}")

    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df["description"] = df["description"].fillna("Unknown Transaction").astype(str).str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["date"] = df["date"].fillna(datetime.now().strftime("%Y-%m-%d"))

    records = []
    for idx, row in df.iterrows():
        raw_amount = float(row["amount"])
        desc_val = str(row["description"])
        
        if is_income_transaction(desc_val, raw_amount):
            final_amount = abs(raw_amount)
        else:
            final_amount = -abs(raw_amount)

        records.append({
            "id": idx + 1,
            "date": row["date"],
            "description": desc_val,
            "amount": final_amount,
            "category": categorize_transaction(desc_val, final_amount)
        })
    return records

# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------
class GoalCreate(BaseModel):
    name: str
    target: float
    current: Optional[float] = 0.0

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "FinPilot API is active."}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global TRANSACTIONS_DB
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")

    contents = await file.read()
    df = None
    encodings = ["utf-8", "latin-1", "cp1252"]
    
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding=enc, sep=None, engine="python")
            break
        except Exception:
            continue

    if df is None:
        raise HTTPException(status_code=400, detail="Could not parse CSV file.")

    try:
        TRANSACTIONS_DB = process_csv_dataframe(df)
        return {"status": "success", "imported_records": len(TRANSACTIONS_DB)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")

@app.get("/dashboard")
def get_dashboard_summary():
    if not TRANSACTIONS_DB:
        return {"total_income": 0, "total_spending": 0, "net_cash_flow": 0, "spending_by_category": {}}

    df = pd.DataFrame(TRANSACTIONS_DB)
    income = float(df[df["amount"] > 0]["amount"].sum())
    spending = float(abs(df[df["amount"] < 0]["amount"].sum()))
    net = income - spending

    spending_df = df[df["amount"] < 0].copy()
    if not spending_df.empty:
        spending_df["abs_amount"] = spending_df["amount"].abs()
        category_summary = spending_df.groupby("category")["abs_amount"].sum().round(2).to_dict()
    else:
        category_summary = {}

    return {
        "total_income": round(income, 2),
        "total_spending": round(spending, 2),
        "net_cash_flow": round(net, 2),
        "spending_by_category": category_summary
    }

@app.get("/recurring")
def get_recurring_payments():
    if not TRANSACTIONS_DB:
        return {"recurring_payments": []}

    df = pd.DataFrame(TRANSACTIONS_DB)
    expenses = df[df["amount"] < 0]

    grouped = expenses.groupby(["description", "amount"]).size().reset_index(name="count")
    recurring = grouped[grouped["count"] > 1]

    results = [
        {
            "description": row["description"],
            "amount": abs(float(row["amount"])),
            "frequency_detected": int(row["count"])
        }
        for _, row in recurring.iterrows()
    ]

    return {"recurring_payments": results}

@app.get("/goals")
def get_goals():
    return {"goals": GOALS_DB}

@app.post("/goals")
def create_goal(goal: GoalCreate):
    new_id = len(GOALS_DB) + 1
    new_goal = {"id": new_id, "name": goal.name, "target": goal.target, "current": goal.current}
    GOALS_DB.append(new_goal)
    return {"status": "created", "goal": new_goal}

@app.post("/query")
def query_financial_data(payload: QueryRequest):
    """Flexible NLP endpoint to answer plain English questions."""
    if not TRANSACTIONS_DB:
        return {"answer": "No transactions loaded. Please upload a CSV first."}

    q = payload.query.strip().lower()
    df = pd.DataFrame(TRANSACTIONS_DB)
    expenses_df = df[df["amount"] < 0].copy()
    expenses_df["abs_amount"] = expenses_df["amount"].abs()

    # 1. Largest / Biggest / Highest purchase
    if any(w in q for w in ["largest", "biggest", "highest", "most expensive", "max"]):
        if expenses_df.empty:
            return {"answer": "No expenses found in your dataset."}
        max_row = expenses_df.loc[expenses_df["abs_amount"].idxmax()]
        return {"answer": f"Your largest purchase was ${max_row['abs_amount']:.2f} at '{max_row['description']}' on {max_row['date']}."}

    # 2. Smallest / Lowest / Cheapest purchase
    if any(w in q for w in ["smallest", "lowest", "cheapest", "least", "min"]):
        if expenses_df.empty:
            return {"answer": "No expenses found in your dataset."}
        min_row = expenses_df.loc[expenses_df["abs_amount"].idxmin()]
        return {"answer": f"Your smallest purchase was ${min_row['abs_amount']:.2f} at '{min_row['description']}' on {min_row['date']}."}

    # 3. Total income query
    if any(w in q for w in ["income", "earnings", "earned", "made"]):
        total_income = df[df["amount"] > 0]["amount"].sum()
        return {"answer": f"Your total recorded income is ${total_income:.2f}."}

    # 4. Total spending query
    if any(w in q for w in ["total spending", "total spent", "total expenses", "how much did i spend in total"]):
        total_spent = expenses_df["abs_amount"].sum()
        return {"answer": f"Your total spending across all categories is ${total_spent:.2f}."}

    # 5. Category or Keyword specific spending (e.g., food, groceries, uber, walmart)
    target_categories = []
    
    # Check synonym mapping
    for word, mapped_cats in SYNONYMS.items():
        if word in q:
            target_categories.extend(mapped_cats)

    # Check direct category matches
    for cat in CATEGORY_RULES.keys():
        if cat.lower() in q:
            target_categories.append(cat)

    target_categories = list(set(target_categories))

    # Filter by identified categories or keyword match in description
    if target_categories:
        subset = expenses_df[expenses_df["category"].isin(target_categories)]
        if not subset.empty:
            total = subset["abs_amount"].sum()
            cats_str = ", ".join(target_categories)
            return {"answer": f"You spent ${total:.2f} on {cats_str} across {len(subset)} transaction(s)."}

    # Fallback to general term extraction in description
    search_match = re.search(r"(?:spent|spend|on|for|at)\s+([a-zA-Z0-9\s]+)", q)
    if search_match:
        term = search_match.group(1).strip(" ?")
        # Remove common stop words
        term = re.sub(r"\b(in|total|the|a|an|my)\b", "", term).strip()
        if term:
            subset = expenses_df[
                (expenses_df["category"].str.lower().str.contains(term)) | 
                (expenses_df["description"].str.lower().str.contains(term))
            ]
            if not subset.empty:
                total = subset["abs_amount"].sum()
                return {"answer": f"You spent ${total:.2f} on '{term}' across {len(subset)} transaction(s)."}

    return {
        "answer": "I couldn't find a matching transaction. Try asking:\n"
                  "• 'Which was my smallest purchase?'\n"
                  "• 'How much was spent on food?'\n"
                  "• 'What was my total spending?'"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)