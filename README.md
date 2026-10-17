# DemandLens — Sales Forecasting Application

**From a machine learning model to a deployed web application**

DemandLens is an end-to-end machine learning project that forecasts daily product-family sales for individual stores. It combines a cluster-based LightGBM forecasting engine, a Python FastAPI backend, and a Next.js frontend built with React and TypeScript.

The goal of this project is to demonstrate how to take a traditional machine learning model beyond a notebook and turn it into a working application that users can interact with through a web interface.

The long-term vision is to help demand planners explore business assumptions and compare potential outcomes through a deterministic what-if scenario planner. This repository's current application focuses on the first milestone: **generating and displaying sales forecasts through a deployed web application.**

## Live Application and Resources

| Resource | Link |
|---|---|
| Live application | https://demandlens-api.netlify.app/ |
| GitHub repository | https://github.com/umeaimanMerchant/demandlens |
| FastAPI backend | https://demandlens-api.onrender.com |
| API documentation | https://demandlens-api.onrender.com/docs |
| Health endpoint | https://demandlens-api.onrender.com/health |
| Dataset | [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) |

> **Important:** DemandLens uses historical Favorita sales data from 2013–2017. Forecasts begin on August 16, 2017. This is a machine learning demonstration, not a live retail forecasting service. Predicted sales represent units, not revenue.

---

## 1. The Problem

Retail demand changes with product category, store characteristics, day of the week, seasonality, holidays, promotions, and other factors.

A demand planner may need to answer questions such as:

- How many units might a particular store sell in a product category over the next seven days?
- How does the expected demand pattern change across a 14-day or 30-day horizon?
- Which days are expected to have the highest sales?
- How can historical patterns be used to estimate future demand?
- How could future versions of the application compare different business assumptions?

A machine learning notebook can help answer the first questions, but it does not automatically provide a usable application. Users need a way to submit inputs, receive predictions, and interpret the results.

DemandLens addresses this by connecting four parts:

1. Historical retail data and a trained forecasting model.
2. A Python forecasting engine.
3. A FastAPI service that accepts requests and returns predictions.
4. A web dashboard that lets users request forecasts and explore the results.

## 2. What the Application Does

The current forecasting application allows a user to:

- Select a store.
- Select a product family.
- Choose a forecast horizon of 7, 14, or 30 days.
- Generate daily sales predictions.
- View the forecast in a chart.
- Review summary metrics, including total predicted sales, average daily sales, and the peak forecast day.
- See errors when a request fails or the API returns an unexpected response.

The current frontend sends a zero-promotion assumption to the API. Promotions are therefore treated as an explicit modeling assumption rather than an unknown value silently interpreted as zero.

### Current project scope

**Implemented in the forecasting MVP**

- Historical data preparation and model development.
- Cluster-based LightGBM forecasting.
- Recursive multi-step prediction.
- FastAPI request validation and forecast endpoints.
- Next.js frontend with React, TypeScript, and Recharts.
- Frontend/backend integration.
- Deployment of the frontend and backend.

**Planned for a later phase**

- Editable business assumptions.
- Promotion what-if scenarios.
- Baseline-versus-scenario comparisons.
- A natural-language interface that translates user requests into validated scenario inputs.

The future capabilities should not be confused with functionality already implemented in the current application.

---

## 3. Technology Stack

| Technology | Purpose in DemandLens |
|---|---|
| Python | Model development, data processing, and backend forecasting logic. |
| pandas | Data cleaning, joining datasets, time-series transformations, and feature engineering. |
| NumPy | Numerical operations and array handling. |
| scikit-learn | Supporting machine learning utilities and evaluation workflows. |
| LightGBM | Training the sales forecasting models. |
| joblib | Saving and loading the trained model artifact. |
| FastAPI | Exposing the forecasting engine through HTTP endpoints. |
| Pydantic | Validating incoming API request data. |
| Uvicorn | Running the Python API server. |
| JavaScript | The programming language underlying the browser-side application and Node.js tooling. |
| TypeScript | Adding static type checking to the frontend code. |
| React | Building the interactive user interface from components and state. |
| Next.js | Providing the frontend application framework and build process. |
| Tailwind CSS | Styling the user interface. |
| Recharts | Displaying forecast results in charts. |
| Git | Tracking source-code changes. |
| GitHub | Hosting the repository and providing the source for deployment. |
| Render | Hosting the Python backend. |
| Netlify | Building and hosting the frontend as a static website. |

### How the technologies work together

Python and LightGBM handle forecasting. FastAPI exposes that functionality through an API. The frontend does not train or run the LightGBM model in the browser. Instead, it sends a request to the backend and renders the returned JSON response.

Next.js and React are not alternatives to JavaScript: React is the UI library, Next.js is the framework, and TypeScript is the language used to write the frontend code.

---

## 4. System Architecture

The application separates the user interface from the machine learning service.

```text
                    USER
                     |
                     v
          Next.js / React Frontend
             Hosted on Netlify
                     |
             HTTP POST /forecast
                     |
                     v
             FastAPI Backend
              Hosted on Render
                     |
              Pydantic validation
                     |
                     v
          Forecasting Python Module
              forecasting.py
                     |
                     v
         Saved LightGBM Model Artifact
       Cluster Models + Historical Data
                     |
                     v
          Recursive Daily Predictions
                     |
                     v
               JSON Response
                     |
                     v
          Frontend Metrics and Chart
```

### Request lifecycle

1. The user selects a store, product family, and forecast horizon.
2. The frontend constructs a JSON request.
3. The request is sent to the backend's `/forecast` endpoint.
4. FastAPI and Pydantic validate the incoming values.
5. The forecasting module selects the relevant cluster model and generates predictions.
6. The backend returns daily predictions as JSON.
7. The frontend validates the basic response structure and updates its state.
8. React renders the chart and summary metrics.

The API is the boundary between the frontend and the model. The frontend does not need to know how LightGBM is trained or how recursive features are calculated.

---

# 5. Follow Along: Build DemandLens Step by Step

This section is intended for students who want to reproduce the project on their own machines.

The examples below use macOS and a terminal. Equivalent commands may be needed on Windows.

## Step 1 — Install and Verify the Prerequisites

You need:

- Git.
- Python.
- Node.js and npm.
- A code editor such as Visual Studio Code.
- A Kaggle account to download the dataset.

Open a terminal and run:

```bash
python3 --version
git --version
node --version
npm --version
```

Each command should print an installed version.

If a command is not found, install the corresponding tool before continuing.

### Why are both Python and Node.js required?

Python runs the machine learning code and backend. Node.js runs the frontend development and build tooling used by Next.js.

You do not need Node.js to train LightGBM, and you do not need Python to render the React interface. They serve different parts of the application.

## Step 2 — Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/umeaimanMerchant/demandlens.git
cd demandlens
```

Inspect the project structure:

```bash
ls
find . -maxdepth 3 -type f
```

The repository contains the major project areas:

```text
demandlens/
├── artifacts/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── forecasting.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── app/
│   │       └── page.tsx
│   ├── package.json
│   └── ...
├── sales-forecasting-model.ipynb
├── .gitignore
└── README.md
```

The exact repository may contain additional configuration files and subdirectories. Inspect the actual files before assuming a file exists.

### Understand the important files

- `sales-forecasting-model.ipynb`: the notebook used for data exploration, feature engineering, training, evaluation, and model artifact creation.
- `artifacts/`: saved model artifacts required for inference.
- `backend/app/forecasting.py`: forecasting logic, including recursive prediction.
- `backend/app/main.py`: FastAPI application, request validation, endpoints, and response handling.
- `backend/requirements.txt`: Python dependencies.
- `frontend/src/app/page.tsx`: the main interactive frontend page.
- `frontend/package.json`: frontend dependencies and scripts.

The repository already includes a model artifact. You can first reproduce the running application using that artifact, then retrain the model as a separate exercise.

**Checkpoint:** You have cloned the repository and can locate the notebook, backend, frontend, and artifacts.

## Step 3 — Understand Git and the Project Environment

If you cloned the repository, you generally do not need to run `git init`. Git is already initialized in the cloned project.

Check the working tree:

```bash
git status --short
```

A `.gitignore` file prevents local environments, generated build files, and private configuration from being committed.

Typical entries include:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
**/.venv/

# Secrets and local configuration
.env
.env.*
!.env.example

# Node / Next.js
node_modules/
.next/
out/

# Operating system
.DS_Store

# Local data
data/raw/
data/processed/
```

Review the existing `.gitignore` before modifying it. Do not replace project-specific rules blindly.

### Important artifact distinction

The saved model artifacts required by the deployed application must be available to the backend. Do not ignore or remove required artifacts simply because they are generated files.

Raw datasets and temporary training outputs can be kept out of Git when appropriate. Large required model artifacts need an intentional storage and deployment strategy.

**Checkpoint:** `git status --short` works, and you understand which files are source code, local data, and required inference artifacts.

## Step 4 — Download the Kaggle Dataset

Open the [Kaggle Store Sales — Time Series Forecasting dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

Download the data files needed by the notebook.

The dataset includes files such as:

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`
- `sample_submission.csv`

Not every file is necessarily required for every stage of the workflow. Use the notebook's data-loading cells to confirm the exact inputs.

Create the local data directory if the notebook expects it:

```bash
mkdir -p data/raw
```

Place the downloaded CSV files inside `data/raw/`.

For example:

```text
data/
└── raw/
    ├── train.csv
    ├── test.csv
    ├── stores.csv
    ├── oil.csv
    ├── holidays_events.csv
    └── transactions.csv
```

The actual list should match the files loaded by your notebook.

### Important: keep the historical timeline consistent

The training data ends on **August 15, 2017**. The current application begins its forecast on **August 16, 2017**.

Do not replace the historical data with unrelated data or change the forecast date without reviewing the feature-generation and inference logic.

**Checkpoint:** The notebook can find the CSV files at the paths it expects.

## Step 5 — Create the Python Environment

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activation is typically:

```powershell
.venv\Scripts\Activate.ps1
```

Check the active Python executable:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

Install the backend dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

The exact package versions should come from the repository's requirements file. Keeping dependencies consistent across development and deployment reduces compatibility problems.

### Verify the core packages

Run:

```bash
python -c "import sys, pandas, numpy, lightgbm, sklearn, joblib; print('Python:', sys.version); print('Python executable:', sys.executable); print('pandas:', pandas.__version__); print('NumPy:', numpy.__version__); print('LightGBM:', lightgbm.__version__); print('scikit-learn:', sklearn.__version__); print('joblib:', joblib.__version__)"
```

This confirms that the main libraries can be imported and reports their versions.

If an import fails, check that the virtual environment is activated and that the installation completed successfully.

### Python version compatibility

A dependency can work with one Python version and fail with another. During development, Render initially selected Python 3.14.3, which created compatibility concerns with the project's older pinned dependencies.

The deployment configuration was adjusted to use Python 3.12.2.

For local reproduction, use a Python version supported by the actual dependencies in `backend/requirements.txt`. Do not assume that the newest Python version is automatically compatible.

**Checkpoint:** The Python environment is active and the core ML libraries import successfully.

## Step 6 — Explore the Forecasting Notebook

Open:

```text
sales-forecasting-model.ipynb
```

Use Jupyter Notebook, JupyterLab, or a compatible notebook environment.

Run the notebook in order, checking the output at each major stage.

### 6.1 Audit the data

Inspect:

- Dataset shapes and column names.
- Date ranges.
- Store and product-family counts.
- Duplicate observations.
- Missing values.
- Negative sales and zero-sales observations.
- Sales distributions and patterns by day of the week.

The training dataset contains approximately 3 million observations and covers 2013 through August 15, 2017.

A data audit is important because forecasting performance depends on correct dates, joins, feature construction, and train/validation boundaries.

### 6.2 Establish a baseline

Before training a complex model, compare it against a simple forecasting approach.

A seasonal-naive baseline predicts sales using a corresponding historical seasonal observation, such as the same day of the week from the previous week.

In the project's initial 16-day validation comparison:

| Model | WAPE |
|---|---:|
| Seasonal-naive baseline | Approximately 20.67% |
| Selected cluster-based LightGBM model | Approximately 15.13% |

WAPE is the weighted absolute percentage error:

\[
\mathrm{WAPE} =
\frac{\sum_i |y_i-\hat y_i|}
{\sum_i |y_i|}\times 100
\]

Lower WAPE indicates lower aggregate absolute error relative to total actual sales, provided the denominator is nonzero.

These are historical validation results from the project, not a guarantee of performance on other datasets or future retail periods.

### 6.3 Build time-series features

The notebook prepares features that help the model identify demand patterns, including:

- Historical sales lags.
- Rolling statistics.
- Calendar and day-of-week features.
- Store and product-family information.
- Holiday information.
- Promotion-related features.

Time-series feature engineering requires special care: a training row must not use information that would have been unavailable at the time of prediction.

For validation, use a chronological split. Do not randomly shuffle time-series observations in a way that allows future information to leak into training.

### 6.4 Train and compare model strategies

The notebook evaluates different modeling approaches. The selected approach is a cluster-based LightGBM strategy.

The final validation results recorded for the project were:

| Approach | Validation WAPE |
|---|---:|
| Cluster-based LightGBM | 15.1316% |
| Store-level LightGBM | 15.3437% |
| Global LightGBM | 16.6511% |
| Store-family LightGBM | 18.5657% |

The cluster model performed best among these evaluated approaches in the project's validation setup.

The selection was based on measured validation performance, not an assumption that clustering is always better.

### 6.5 Save the trained artifact

The notebook produces the model artifact used by the backend.

The repository already contains an artifact, so **retraining is optional for students who only want to run the application**.

If you choose to retrain:

1. Run the notebook's preprocessing and feature-engineering steps.
2. Train the models.
3. Evaluate them using the intended validation window.
4. Save the artifact in the location and format expected by the backend.
5. Confirm that the artifact's structure matches the inference code.

Do not overwrite the working artifact until you have verified that the newly generated artifact can be loaded by the backend.

**Checkpoint:** You understand the dataset, the baseline, the model comparison, and the purpose of the saved artifact.

## Step 7 — Understand Recursive Forecasting

Open:

```text
backend/app/forecasting.py
```

This module contains the logic used to generate forecasts from the saved model artifact.

A multi-step forecast is different from making one independent prediction.

For example, suppose the model needs the previous seven days of sales to predict the next day.

1. Predict sales for August 16.
2. Add that prediction to the working history.
3. Use the updated history to create the lag and rolling features for August 17.
4. Predict August 17.
5. Continue until the requested horizon is complete.

This is called **recursive forecasting**.

The challenge is that actual future sales are unavailable during inference. Earlier predictions therefore become inputs for later predictions.

### Why this matters

- Prediction errors can accumulate over a longer horizon.
- Feature generation must follow the same rules used during training.
- The model must not accidentally use actual future sales.
- The working history must be updated in the correct order.
- The model must receive the features in the format it expects.

The forecasting module also uses the saved cluster models and historical information required for inference.

When troubleshooting, inspect this module before changing the API or frontend. A request can be valid and still fail during model selection, feature generation, artifact loading, or prediction.

**Checkpoint:** You can explain how the backend generates more than one future day without having actual future sales available.

## Step 8 — Test the Forecasting Engine Before the API

Before connecting the model to FastAPI, test the forecasting function independently.

Use the notebook or a small Python test script to:

1. Load the saved artifact.
2. Select a valid store and product family.
3. Request a seven-day forecast.
4. Confirm that the function returns the expected number of dates.
5. Check that predicted sales are numeric and non-negative.
6. Verify that the forecast dates are consecutive and begin on the requested date.
7. Repeat with 14-day and 30-day horizons.

The forecast should contain one result per requested day.

Do not assume that a successful model load proves that inference works. Artifact loading, feature generation, model selection, and recursive prediction all need to be exercised.

**Checkpoint:** The forecasting function works independently of HTTP and the frontend.

## Step 9 — Understand and Run the FastAPI Backend

Open:

```text
backend/app/main.py
```

This file connects the forecasting engine to HTTP requests.

It is responsible for:

- Creating the FastAPI application.
- Loading the saved artifact during application startup.
- Configuring Cross-Origin Resource Sharing (CORS).
- Validating forecast request data with Pydantic.
- Exposing health and forecast endpoints.
- Calling the forecasting function.
- Returning predictions as JSON.
- Handling expected request and forecasting errors.

The actual forecasting algorithm belongs in `forecasting.py`; `main.py` is the service layer around it.

### 9.1 Understand request validation

The forecast API validates inputs such as:

- Store number.
- Product family.
- Start date.
- Forecast horizon.
- Promotion assumptions.

The supported horizons are 7, 14, and 30 days.

The API also requires the promotion assumption to be explicit: the caller must provide a promotion list or explicitly request that promotions be assumed to be zero. When a promotion list is provided, its length must match the forecast horizon.

This prevents a missing input from silently being treated as a business assumption.

### 9.2 Start the API locally

From the repository root, with the virtual environment activated, run:

```bash
uvicorn backend.app.main:app --reload
```

If the application starts successfully, open:

- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

The health endpoint helps confirm that the service is responding. It does not necessarily prove that a full forecast can be generated successfully.

### 9.3 Test the forecast endpoint

In the interactive API documentation, submit a valid request to `POST /forecast` using the schema displayed by the running application.

The current frontend sends a request equivalent to:

```json
{
  "store_nbr": 1,
  "family": "BEVERAGES",
  "start_date": "2017-08-16",
  "horizon": 7,
  "assume_zero_promotions": true
}
```

This example reflects the current frontend's request shape. Confirm the schema in `/docs` if you have changed the backend.

A successful response contains a `forecast` collection with one record per requested date. Each daily record includes the date, store, family, promotion assumption, and predicted sales.

### 9.4 Understand HTTP errors

The backend uses HTTP status codes to distinguish successful requests from rejected or failed requests.

For example:

- `200`: successful response.
- `422`: request validation failed or a handled input-related error was returned.
- `500` or `502`: may indicate a server-side or hosting/proxy problem and requires further investigation.

Read the response body and backend logs instead of assuming every failure comes from the frontend.

**Checkpoint:** `/docs` loads, `/health` responds, and `POST /forecast` returns a valid forecast locally.

## Step 10 — Set Up the Next.js Frontend

Open a second terminal and move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Inspect the available scripts in `package.json`.

### 10.1 Configure the backend URL

The frontend reads its backend URL from:

```text
NEXT_PUBLIC_API_URL
```

For local development, create `frontend/.env.local` with:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Do not commit local environment files containing private values.

The frontend's API URL configuration removes trailing slashes, which helps prevent malformed URLs when the environment variable ends with `/`.

### 10.2 Start the frontend

Run:

```bash
npm run dev
```

Open the local URL printed by Next.js, typically:

```text
http://localhost:3000
```

The API and frontend must both be running when you test a forecast locally.

### 10.3 Understand the main frontend file

Open:

```text
frontend/src/app/page.tsx
```

This file contains the main interactive dashboard.

The important responsibilities are:

- Holding the selected store, family, and horizon in React state.
- Sending the forecast request to FastAPI.
- Showing loading and error states.
- Checking that the response contains the expected forecast array and number of days.
- Calculating summary metrics from returned predictions.
- Rendering the chart and metrics.

The frontend uses TypeScript types to describe the expected response structure. However, a TypeScript type does not validate JSON at runtime. That is why the application also checks the returned forecast structure.

The chart is rendered with Recharts. The frontend displays the results returned by the backend; it does not retrain the model or independently calculate the model's predictions.

### 10.4 Understand the API request flow

When the user clicks the forecast button:

1. The UI enters a loading state.
2. The old forecast and error message are cleared.
3. The frontend sends a JSON request to `/forecast`.
4. The response body is parsed.
5. An unsuccessful HTTP response is converted into a readable error.
6. The forecast response is checked for the expected array and horizon length.
7. Valid predictions are stored in React state.
8. React updates the chart and summary metrics.
9. The loading state is reset even if the request fails.

The `fetch()` API does not automatically throw for every HTTP error status. Checking `response.ok` is therefore an important part of reliable request handling.

**Checkpoint:** The frontend builds and displays forecasts returned by the locally running API.

## Step 11 — Test the Complete Application Locally

Before deploying, test the entire request path.

Keep the backend and frontend running in separate terminals.

### Functional checklist

- [ ] The frontend loads without a compilation error.
- [ ] The API health endpoint responds.
- [ ] The store selection works.
- [ ] The product-family selection works.
- [ ] A seven-day forecast succeeds.
- [ ] A 14-day forecast succeeds.
- [ ] A 30-day forecast succeeds.
- [ ] The returned number of days matches the requested horizon.
- [ ] Dates are consecutive and begin on the configured start date.
- [ ] The chart displays the returned predictions.
- [ ] Summary metrics match the displayed forecast values.
- [ ] An invalid request produces a useful error.
- [ ] Loading ends after both successful and failed requests.

Do not deploy based only on the fact that the homepage loads. A working homepage does not prove that the model can generate a forecast.

## Step 12 — Build the Frontend for Production

From the `frontend/` directory, run:

```bash
npm run build
```

This checks the production build rather than only the development server.

The application is configured for a static Next.js export. The relevant Next.js configuration includes:

```javascript
output: "export"
```

The configuration also uses trailing slashes for exported routes. Review the actual `next.config.*` file rather than creating a second configuration file.

With static export enabled, a successful build should generate an `out/` directory containing the static website, including `out/index.html`.

The frontend is hosted separately from the Python API. It is not a Python service, and Netlify does not run the LightGBM model.

**Checkpoint:** `npm run build` succeeds and the static output exists.

---

# 6. Deploy the Backend to Render

The backend runs as a Python web service on Render.

Live backend:

https://demandlens-api.onrender.com

## 6.1 Create the Render web service

1. Push the project to GitHub.
2. Sign in to Render.
3. Create a new **Web Service**.
4. Connect the DemandLens GitHub repository.
5. Select the branch containing the working code, currently `main`.
6. Configure the service using the settings below.

### Render configuration

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Python |
| Branch | `main` |
| Root directory | Leave blank |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` |
| Python version | `3.12.2`, using `PYTHON_VERSION=3.12.2` if needed |

The root directory is intentionally blank because the start command and artifact paths expect the repository root to be the working directory. Setting the root directory to `backend` without changing the commands and paths can break imports or artifact discovery.

### Why the start command matters

The command:

```bash
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

does four important things:

- `PYTHONPATH=.` makes the repository root available for Python imports.
- `backend.app.main:app` tells Uvicorn where to find the FastAPI application object.
- `--host 0.0.0.0` makes the service reachable through the hosting environment.
- `--port $PORT` uses the port assigned by Render.

Do not hardcode port `8000` in the production start command.

### 6.2 Verify artifact paths

The backend needs to find the saved model artifact at runtime.

Because the deployment uses the repository root as the working directory, keep the artifact paths and repository layout consistent with the code. Check that the required artifact files are committed to the repository or otherwise made available to the service.

If the artifact is missing, incorrectly named, or saved in a different directory, the service may fail during startup or forecasting.

### 6.3 Configure the Python version

During deployment, Render initially selected Python 3.14.3. This was incompatible with the project's dependency setup.

Use the intended compatible version:

```text
PYTHON_VERSION=3.12.2
```

Add it as a service environment variable if that is how the Render service is configured. If the build still uses cached dependencies from an earlier attempt, review the build logs and use Render's cache-clearing redeploy option when appropriate.

### 6.4 Verify the deployed service

After deployment, check:

- https://demandlens-api.onrender.com/health
- https://demandlens-api.onrender.com/docs

Then test `POST /forecast`.

A successful health check confirms that the health endpoint is reachable. It does not guarantee that the complete forecasting path works. Always test an actual forecast.

### 6.5 Configure CORS

CORS controls which browser origins are allowed to call the backend.

The FastAPI application must allow the deployed Netlify frontend origin. Review the CORS middleware in `backend/app/main.py` and ensure it includes the actual production frontend URL.

For local development, the local frontend origin also needs to be allowed.

CORS is a browser security policy; it is not authentication. Allow only the origins required by the application rather than using a permissive configuration without a reason.

**Checkpoint:** Render starts the API, `/health` and `/docs` respond, and a real forecast request succeeds.

---

# 7. Deploy the Frontend to Netlify

The frontend is built as a static website and hosted on Netlify.

Live frontend:

https://demandlens-api.netlify.app/

## 7.1 Create the Netlify site

1. Sign in to Netlify.
2. Import the DemandLens repository from GitHub.
3. Select the branch containing the working frontend.
4. Configure the build settings below.
5. Add the production environment variable before building.

### Netlify build configuration

| Setting | Value |
|---|---|
| Base directory | `frontend` |
| Build command | `npm run build` |
| Publish directory | `out`, relative to the base directory |

The effective output directory is:

```text
frontend/out/
```

Netlify must publish the generated static export, not Next.js's internal `.next/` directory.

## 7.2 Set the frontend API URL

In the Netlify site's environment-variable settings, add:

```dotenv
NEXT_PUBLIC_API_URL=https://demandlens-api.onrender.com
```

Use the actual deployed backend URL.

The `NEXT_PUBLIC_` prefix is important because Next.js exposes this value to the browser-side application. The value is incorporated into the frontend during the build.

**After changing this variable, trigger a new deployment.** Changing the environment variable alone does not update an already-built static website.

Do not use `http://localhost:8000` in the deployed frontend. In a visitor's browser, `localhost` refers to that visitor's own computer, not the Render backend.

## 7.3 Confirm the static export

After the build completes, verify that the published output contains `index.html`.

The project uses:

```javascript
output: "export"
```

and trailing-slash routing in its Next.js configuration to support static hosting.

If Netlify reports a successful build but returns a 404 for the homepage, first check the publish directory and whether `frontend/out/index.html` was generated.

## 7.4 Test the live application

Open the Netlify URL and run forecasts for each supported horizon.

Use browser developer tools and the Render logs if a request fails. Check the request URL, response status, response body, and backend logs before changing code.

**Checkpoint:** The live Netlify application calls the live Render API and displays the returned forecast.

---

# 8. End-to-End Verification Checklist

Use this checklist before considering the current forecasting MVP reproducible.

## Data and model

- [ ] Dataset files are available at the expected paths.
- [ ] The notebook can load the data.
- [ ] The historical cutoff is consistent with the inference setup.
- [ ] The saved artifact exists and can be loaded.
- [ ] The model produces numeric predictions.
- [ ] Recursive features are generated without future-data leakage.

## Backend

- [ ] Dependencies install in the intended Python environment.
- [ ] FastAPI starts successfully.
- [ ] `/health` responds.
- [ ] `/docs` loads.
- [ ] Valid requests pass Pydantic validation.
- [ ] Invalid requests return useful errors.
- [ ] `POST /forecast` succeeds for 7, 14, and 30 days.
- [ ] The response contains one daily result per forecast date.

## Frontend

- [ ] `npm install` completes.
- [ ] `npm run dev` works locally.
- [ ] `npm run build` succeeds.
- [ ] `NEXT_PUBLIC_API_URL` points to the correct API.
- [ ] The frontend handles loading, success, and error states.
- [ ] The chart and summary metrics match the API response.

## Deployment

- [ ] Render uses the correct Python version and start command.
- [ ] Required artifacts are available to Render.
- [ ] CORS allows the deployed Netlify origin.
- [ ] Netlify publishes `frontend/out`.
- [ ] Netlify's API environment variable points to Render.
- [ ] The live application generates a real forecast.

---

# 9. Engineering Decisions: Options, Trade-offs, and Final Choices

A major goal of DemandLens is to learn not only how to write code, but also how to make engineering decisions. Each choice should solve a specific problem and be evaluated against alternatives.

## Decision 1 — Which forecasting model architecture should we use?

**Problem:** How should the model share information across stores and product families?

| Option | Advantages | Trade-offs |
|---|---|---|
| Global model | One model is easier to manage and deploy; can share information across many observations. | A single model may struggle to represent all store-specific and product-specific patterns. |
| Store-level models | Can learn store-specific demand behavior. | More models to train, save, load, and maintain; some stores have less data. |
| Store-family models | Can specialize in a specific store and product family. | Many model combinations, more artifacts, and potentially less data for each model. |
| Cluster-based models | Share information across groups of stores while allowing differences between clusters. | Requires cluster assignment and additional inference logic. |

**Final choice: cluster-based LightGBM models.**

The recorded validation WAPE was approximately 15.13% for the cluster-based approach, compared with 15.34% for store-level, 16.65% for global, and 18.57% for store-family models.

The cluster-based approach was selected because it performed best among the evaluated alternatives in the project's validation setup while retaining a manageable number of models.

The final artifact contains 17 cluster models.

**Lesson:** Choose the modeling architecture based on a meaningful validation design and operational constraints, not just model complexity.

## Decision 2 — Recursive forecasting or direct multi-step prediction?

**Option A: recursive forecasting**

Predict one future day, add that prediction to the working history, and use the updated history to predict the next day.

- Reuses a one-step model.
- Supports multiple forecast horizons.
- Requires careful feature updates.
- Can accumulate prediction errors over time.

**Option B: direct forecasting**

Train separate models or output structures for specific future horizons.

- Avoids feeding every previous prediction back into the next prediction.
- Can specialize in particular horizons.
- May require more models or a more complicated training pipeline.
- Requires a separate approach to generating the requested sequence of future predictions.

**Final choice: recursive forecasting.**

The project uses recursive predictions so that future lag and rolling features can be generated when actual future sales are unavailable.

**Lesson:** Recursive forecasting simplifies some parts of the modeling pipeline, but its inference logic must match the training feature logic. Longer-horizon performance must be evaluated rather than assumed.

## Decision 3 — Should the model run in the browser or on the backend?

**Option A: run inference in the browser**

This could be appropriate for a model designed for client-side inference, but it would require a compatible model runtime and a way to distribute the model and its dependencies.

**Option B: run inference in Python on the backend**

The backend can load the existing LightGBM artifact and use the same Python-based feature-engineering logic used during model development.

**Final choice: Python inference through FastAPI.**

The browser sends inputs to the API, and the backend returns predictions as JSON.

**Lesson:** Keep model inference close to its existing runtime and separate the user interface from model-specific logic.

## Decision 4 — Why separate the frontend and backend?

**Option A: one combined application**

A single application can reduce the number of services to configure, but mixing the user interface and Python inference may complicate deployment and maintenance.

**Option B: independent frontend and backend services**

The frontend and backend can be built, deployed, and debugged independently. The trade-offs include CORS configuration, API URL management, and separate deployment settings.

**Final choice: a Next.js frontend and a FastAPI backend deployed as separate services.**

Netlify hosts the frontend, and Render hosts the Python backend.

**Lesson:** Separation creates a clean interface between systems, but it introduces operational responsibilities. Both services must be configured to communicate with each other.

## Decision 5 — How should API inputs be validated?

**Option A: trust the frontend**

The frontend can restrict the options a user selects, but API requests can still be sent directly or constructed incorrectly.

**Option B: validate requests on the backend**

Pydantic validates incoming requests independently of the user interface.

**Final choice: validate at the API boundary with Pydantic.**

The backend validates fields such as the store number, product family, date, supported forecast horizon, and promotion assumptions.

The backend is the source of truth for the request contract.

**Lesson:** Frontend validation improves usability; backend validation protects the API from invalid input regardless of where the request originates.

## Decision 6 — How should missing future promotion information be handled?

**Option A: silently treat missing promotions as zero**

This is simple but can hide an important modeling assumption.

**Option B: require explicit promotion information or an explicit zero-promotion assumption**

This makes the caller's intent clear and helps prevent accidental assumptions.

**Final choice: make the promotion assumption explicit.**

The API accepts promotion data or an explicit instruction to assume zero promotions. The current frontend uses the zero-promotion option.

**Lesson:** Missing information and an explicit business assumption are not the same thing. Future scenario planning should preserve this distinction.

## Decision 7 — How should the model artifact and historical data be handled?

**Option A: retrain the model on every API request**

This would increase latency and resource usage and would mix training responsibilities with inference.

**Option B: load a saved artifact and use it for inference**

The backend loads the artifact during startup and reuses it for forecast requests.

**Final choice: use a saved joblib artifact containing the models and the historical information required by the inference logic.**

The artifact makes deployment possible without running the entire training notebook each time the service starts or receives a request.

**Trade-offs:** The artifact must be available at the expected path, compatible with the installed Python packages, and small enough to fit within the hosting service's memory constraints.

**Lesson:** A model is not just a model file. The full inference artifact may include supporting history, metadata, and assumptions that the forecasting function depends on.

## Decision 8 — How should the frontend handle API requests and responses?

**Option A: display whatever the API returns**

This is quick to implement but makes the frontend fragile when the API fails or returns unexpected data.

**Option B: validate the HTTP status and basic response structure**

The frontend can display useful errors and avoid rendering malformed results.

**Final choice: explicitly check `response.ok`, handle error responses, verify the forecast array and expected horizon length, and reset loading state in `finally`.**

The TypeScript response type improves development-time checking, while the runtime checks protect against unexpected API responses.

**Lesson:** Static types do not replace runtime validation for data received over a network.

## Decision 9 — Why Netlify and Render instead of AWS?

**Option A: deploy the frontend and backend using AWS infrastructure**

An AWS deployment could use services such as S3 and CloudFront for static hosting, and a container-based service for the backend. It can provide more infrastructure control but requires additional setup and operational knowledge.

**Option B: use managed hosting services**

Netlify and Render provide a simpler path to deploying the frontend and Python API independently.

**Final choice: Netlify for the frontend and Render for the backend.**

This kept the first deployment focused on getting the complete ML application working instead of first learning a larger infrastructure stack.

**Trade-offs:** Managed hosting reduces initial setup, but configuration, build behavior, resource limits, environment variables, logs, and service availability still need to be understood.

**Lesson:** Pick deployment infrastructure that fits the project's current learning and operational goals. A simpler deployment is still a real deployment, with real engineering responsibilities.

## Decision 10 — Historical forecast or live retail forecast?

**Option A: train and maintain a continuously updated model**

This would require an ongoing data pipeline, scheduled retraining or refresh logic, monitoring, and access to new retail data.

**Option B: use a fixed historical dataset to demonstrate the full application workflow**

This makes the project reproducible and keeps the focus on machine learning, API development, frontend integration, and deployment.

**Final choice: use the historical Favorita dataset for the current MVP.**

The forecast starts on August 16, 2017, after the training cutoff.

**Lesson:** Deploying a model does not make its data current. A live forecasting system requires a separate strategy for acquiring new data, validating it, and updating the model or historical state.

## Decision 11 — Should the first version include an LLM?

**Option A: add a natural-language interface immediately**

An LLM could help users express questions in natural language, but it would add another component before the core forecasting workflow was fully established.

**Option B: first build a deterministic forecasting API and user interface**

This creates a clear, testable contract for inputs, assumptions, and predictions.

**Final choice: build the deterministic forecasting workflow first.**

A natural-language interface remains a possible future enhancement, after the underlying scenario workflow has a well-defined request contract and reliable validation.

**Lesson:** Establish a reliable deterministic foundation before adding an LLM to interpret user requests or orchestrate business actions.

---

# 10. Errors Encountered During Development and What to Check

The current code and deployment configuration incorporate fixes for several issues encountered during development. This section documents the failure patterns so that students can diagnose them if they reproduce an earlier state, change the configuration, or deploy to a new environment.

## Issue 1 — Python dependency installation failed on Render

**Observed problem:** Render initially selected Python 3.14.3, and the pinned dependency set was not compatible with that environment.

**Resolution:** Configure the service to use the intended compatible Python version, 3.12.2, and redeploy.

**Where to investigate:**

- Render build logs.
- `backend/requirements.txt`.
- The configured Python version.
- Package compatibility with the selected interpreter.

**Lesson:** A dependency file alone does not guarantee a reproducible environment. The Python version is part of the deployment configuration.

## Issue 2 — Netlify homepage returned 404

**Observed problem:** An early Netlify deployment reported no published files or returned a 404.

**Cause:** The publish directory did not point to the static export generated by Next.js.

**Resolution:** Enable static export using `output: "export"`, retain the intended trailing-slash configuration, and set the publish directory to `out` relative to the `frontend` base directory.

The expected generated homepage is:

```text
frontend/out/index.html
```

**Where to investigate:**

- `frontend/next.config.*`.
- Netlify's base directory.
- Netlify's build command.
- Netlify's publish directory.
- Whether the build actually generated `out/index.html`.

**Lesson:** A successful build is not enough if the hosting platform publishes the wrong directory.

## Issue 3 — The deployed frontend was missing Tailwind styling

**Observed problem:** The deployed frontend initially did not display the expected CSS classes.

**Cause:** Relevant PostCSS configuration and package changes were present locally but had not been pushed to the GitHub repository used for deployment.

**Resolution:** Include the required Tailwind/PostCSS configuration and dependency changes in Git, then commit, push, and redeploy.

The relevant configuration included `postcss.config.mjs` and the required `@tailwindcss/postcss` package.

**Where to investigate:**

- `frontend/package.json`.
- The package lock file.
- `frontend/postcss.config.mjs`.
- The CSS entry point and Tailwind setup.
- Whether the latest changes were actually pushed to the deployed branch.
- The Netlify build logs.

**Lesson:** A local application can look correct while the deployed build is missing configuration files or dependencies.

## Issue 4 — The live frontend tried to call localhost

**Observed problem:** The deployed frontend loaded, but forecasting failed because the API URL defaulted to `http://localhost:8000`.

**Cause:** The frontend was built without the correct production value for `NEXT_PUBLIC_API_URL`.

**Resolution:** Configure the Netlify environment variable:

```dotenv
NEXT_PUBLIC_API_URL=https://demandlens-api.onrender.com
```

Then trigger a fresh build and deployment.

**Where to investigate:**

- The value of `NEXT_PUBLIC_API_URL`.
- The URL shown in the browser's Network tab.
- Whether the latest build contains the correct API URL.
- The Render service's status and logs.

**Lesson:** In a browser, `localhost` points to the visitor's computer. The deployed frontend must use the public backend URL.

## Issue 5 — The backend worked, but `POST /forecast` returned 502

**Observed problem:** During development, the health endpoint, API documentation, and CORS preflight request worked, but a direct forecast request returned HTTP 502 with an empty response body.

**What this told us:** The service could respond to basic requests, but the complete forecast path was failing. This narrowed the investigation to the forecast request, its processing, or the hosting environment.

The exact root cause was not conclusively established in the discussion available for this README. Memory pressure was considered a possible cause because the inference artifact includes approximately 3 million historical rows and the service was running with a 512 MB memory limit.

**Where to investigate if this recurs:**

1. Read the Render logs for the exact time of the failed request.
2. Confirm that the artifact loads successfully at startup.
3. Inspect memory usage during a real forecast request.
4. Check whether the failure occurs during artifact loading, cluster selection, feature generation, or recursive prediction.
5. Verify that the requested store, family, horizon, and promotion assumptions pass validation.
6. Reproduce the same request locally.
7. Compare a short forecast with longer horizons.
8. Check the hosting service's resource limits and restart history.

**Lesson:** A successful health check does not prove that a computationally expensive endpoint works. Test the complete inference path and use logs and resource metrics before assigning a cause.

The current code and deployment setup have been updated since the earlier debugging work. If the error appears again, diagnose the current logs rather than assuming that the historical issue has the same cause.

## Issue 6 — CORS or cross-origin requests failed

**Observed problem:** The frontend and backend run on different origins, so the browser requires the API to permit the frontend's origin.

**Resolution:** Configure FastAPI's CORS middleware to include the deployed Netlify origin and the local development origin where needed.

**Where to investigate:**

- The CORS middleware in `backend/app/main.py`.
- The browser console.
- The Network tab's `OPTIONS` and `POST` requests.
- Whether the frontend is calling the expected backend URL.

**Lesson:** CORS and backend availability are separate concerns. A request can reach a running service and still be blocked by the browser's cross-origin policy.

## Issue 7 — The model artifact could not be found or loaded

**Possible symptoms:** Startup failures, missing-file errors, import errors, or failures when the forecast function attempts to use the model.

**Where to investigate:**

- The artifact's actual location in the repository.
- The path construction in `backend/app/main.py`.
- The model-loading logic.
- The Render root directory.
- Whether the required artifact files were committed or otherwise included in deployment.
- Python and package compatibility.

**Prevention:** Keep the expected artifact structure and the backend's path logic aligned. Test a clean startup and a real forecast after changing the artifact or repository layout.

## Issue 8 — The API returned a validation error

**Possible symptoms:** HTTP 422 or an error message describing an invalid request.

**Where to investigate:**

- Store number and product-family values.
- Supported forecast horizons.
- Date formatting.
- Promotion assumptions.
- Promotion-list length if explicit promotions are used.
- The request schema shown in `/docs`.

**Prevention:** Treat the Pydantic schema as the API contract. When the frontend request changes, verify that the backend schema and forecasting function agree.

## Issue 9 — The frontend displayed an error or unexpected response

**Possible symptoms:** The request finishes, but no forecast appears; the chart does not update; or an error is displayed.

**Where to investigate:**

- The request URL and HTTP status.
- The response body.
- The `response.ok` check.
- The shape of the returned `forecast` array.
- Whether the returned number of daily records matches the selected horizon.
- The browser console and Render logs.

**Prevention:** Keep the frontend's expected response shape aligned with the backend response model. TypeScript types are useful, but incoming JSON still requires runtime checks.

---

# 11. Troubleshooting Order

When something fails, investigate the system in this order rather than changing multiple parts at once.

1. **Environment:** Are the correct Python and Node.js versions active?
2. **Dependencies:** Do package installations complete without errors?
3. **Data and artifact:** Can the notebook load its inputs, and can the backend load the artifact?
4. **Forecasting function:** Can the model generate a valid forecast without FastAPI?
5. **Backend:** Does the API start, and does `POST /forecast` succeed locally?
6. **Frontend:** Does the frontend build and call the correct local API URL?
7. **Deployment configuration:** Are Render's Python version, commands, working directory, and artifact paths correct?
8. **Static hosting:** Does Netlify publish the generated `frontend/out` directory?
9. **Environment variables:** Does the deployed frontend contain the correct Render API URL?
10. **Cross-origin requests:** Does the backend permit the deployed frontend origin?
11. **Production inference:** Does a real forecast succeed on Render, with logs and resource usage checked?

Change one thing at a time and retest the affected layer. This makes it easier to identify the actual cause of a failure.

---

# 12. Project Roadmap

## Phase 1 — Forecasting MVP

**Status: implemented**

- Audit the historical dataset.
- Establish a baseline.
- Engineer time-series features.
- Compare model architectures.
- Train and save the selected LightGBM models.
- Implement recursive forecasting.
- Expose forecasts through FastAPI.
- Build the interactive Next.js dashboard.
- Validate the frontend/backend contract.
- Deploy the frontend and backend.
- Test the end-to-end application.

## Phase 2 — Deterministic What-If Scenario Planner

**Planned**

- Make promotion and other supported business assumptions editable.
- Define a validated scenario request contract.
- Generate a baseline forecast.
- Generate a scenario forecast.
- Compare baseline and scenario outcomes.
- Display the daily and aggregate differences.
- Test the effect of assumptions independently of the frontend.

The scenario engine should be deterministic and testable before introducing a natural-language interface.

## Phase 3 — Natural-Language Interface

**Future direction**

- Allow users to express a scenario request in plain language.
- Translate the request into structured, validated inputs.
- Ask for clarification when required information is missing.
- Use the existing deterministic forecasting workflow to generate results.
- Explain the assumptions behind the output.

An LLM would help interpret user intent; it would not replace the forecasting model or the API's validation rules.

---

# 13. What Students Should Learn from This Project

DemandLens demonstrates that deploying machine learning requires more than selecting an algorithm and reporting a metric.

By following the project, students can learn to:

- Audit a real-world time-series dataset.
- Establish a baseline before training more complex models.
- Compare modeling approaches using a consistent validation setup.
- Engineer leakage-aware forecasting features.
- Implement recursive multi-step prediction.
- Save and load model artifacts.
- Separate model logic from API logic.
- Validate requests with Pydantic.
- Build an interactive frontend using React and TypeScript.
- Connect a frontend to a Python API.
- Handle HTTP errors and unexpected responses.
- Configure environment variables and CORS.
- Build and deploy a static Next.js export.
- Debug build, dependency, routing, and inference problems.
- Make engineering decisions based on trade-offs rather than habit.

The most important lesson is to build a reliable path from input to prediction, test each layer independently, and then verify the complete system after deployment.

## Final Note

DemandLens is a learning and portfolio project focused on applied machine learning and end-to-end application engineering. Its current forecasting workflow uses historical data, saved LightGBM models, recursive inference, FastAPI, and a deployed Next.js dashboard.

The next milestone is to extend that working foundation into a deterministic what-if scenario planner without compromising the reliability and clarity of the existing forecasting API.
