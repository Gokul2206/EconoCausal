```markdown
# EconoCausal: Prescriptive AI & Dynamic Pricing Engine

**EconoCausal** is an enterprise-grade Prescriptive AI platform designed to move beyond traditional predictive machine learning. Using Double/Debiased Machine Learning (DML), this engine calculates the Individual Treatment Effect (ITE) of marketing campaigns to identify "Persuadables"—customers who will only convert *if and only if* they receive a discount.

Rather than wasting capital on customers who would have bought anyway, EconoCausal uses a SciPy-driven Linear Programming solver to mathematically allocate marketing budgets, maximizing revenue lift while adhering to strict financial constraints.

## 🚀 Business Value
* **Causal Uplift:** Isolates true causation from correlation using Microsoft's DoWhy and EconML.
* **Capital Efficiency:** Uses mathematical optimization (Knapsack algorithm) to prescribe the exact dollar amount to offer each user based on their marginal utility per dollar.
* **Risk Mitigation:** Features an automated Kolmogorov-Smirnov (KS) data drift detector to warn stakeholders if real-world customer behavior deviates from the training baseline.

## 🏗️ System Architecture
* **Backend Engine:** Python, FastAPI, SciPy, Pandas, NumPy
* **Causal ML Pipeline:** EconML, DoWhy (Double Machine Learning)
* **Frontend Dashboard:** React.js, Plotly.js, CSS3 (Enterprise Glassmorphism)

## 📊 Key Features & Modules
1. **Causal Graphing (DAG):** Identifies Confounders, Treatments, and Outcomes to eliminate hidden bias in historical observational data.
2. **Uplift & Qini Curves:** Visually proves that the causal model's targeting strategy mathematically outperforms random campaign rollouts.
3. **Prescriptive Optimization:** A SciPy algorithm that ingests thousands of customer ITE scores and a strict budget constraint to generate a mathematically perfect discount allocation matrix.
4. **Data Operations & Drift QA:** A REST API integration that allows non-technical executives to "pull" the latest AI strategy and validate data integrity before spending capital.

## 🛠️ Installation & Setup

### 1. Backend (FastAPI Engine)
Navigate to the backend directory and install the required Python dependencies:
```bash
pip install -r requirements.txt

```

Start the local Uvicorn server:

```bash
uvicorn api.main:app --reload

```

*The API will be available at `http://localhost:8000*`

### 2. Frontend (React Dashboard)

Navigate to the frontend directory and install the Node modules:

```bash
npm install

```

Start the React development server:

```bash
npm start

```

*The dashboard will automatically open at `http://localhost:3000*`

## 🧑‍💻 Usage

1. Open the **Data Operations** tab to pull the latest generated strategy from the Causal ML pipeline.
2. Use the **Quality Assurance** module to upload recent campaign data (`.csv`) to ensure the underlying data distributions have not drifted.
3. Navigate to the **Strategy Simulator**, enter a strict budget constraint, and run the SciPy Optimizer to generate the final campaign rollout matrix.
4. Review the **Executive Summary** for plain-English financial projections and the final customer targeting list.

```

---

### 2. `requirements.txt`
Copy this block and save it as `requirements.txt` in the same folder as your Python backend. This tells the system exactly which libraries are needed to run your engine.

```txt
fastapi==0.104.1
uvicorn==0.24.0.post1
python-multipart==0.0.6
pandas==2.1.3
numpy==1.26.2
scipy==1.11.4
scikit-learn
lightgbm
econml==0.14.1
dowhy==0.10.1
networkx==3.1
matplotlib
plotly
statsmodels
pydot

```