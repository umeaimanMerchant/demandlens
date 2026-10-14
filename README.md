
# DemandLens — Sales Forecasting & What-If Planner

DemandLens is a machine learning web application that forecasts daily product-family sales for individual stores. It combines a trained LightGBM forecasting engine with a FastAPI backend and a Next.js dashboard. The long-term goal is to let users explore how business assumptions, such as promotions, affect expected demand.

## Problem & Approach

The project uses the Kaggle Store Sales — Time Series Forecasting dataset from Favorita, Ecuador.

- **Model:** Cluster-based LightGBM models, selected after comparing cluster, store-level, store-family, and global approaches.
- **Features:** Historical sales lags, rolling statistics, calendar patterns, store and product attributes, holidays, and promotions.
- **Forecasting:** Recursive multi-step prediction, where earlier predictions become inputs for later forecast dates.
- **Output:** Daily predicted unit sales for a selected store, product family, and 7-, 14-, or 30-day horizon.

On a 16-day historical validation window, the cluster model achieved approximately 15.13% WAPE, compared with 20.67% for the seasonal-naive baseline. These are historical validation results, not a guarantee of future performance.

## Architecture

- **Frontend:** Next.js, React, TypeScript, Tailwind CSS, and Recharts.
- **Backend:** Python, FastAPI, Pydantic, and Uvicorn.
- **Machine learning:** LightGBM, pandas, NumPy, and joblib.
- **Model artifact:** Saved cluster models and the historical data required for recursive inference.
- **Deployment target:** Vercel for the frontend and Render for the backend.

The frontend sends validated inputs to the API. The Python backend loads the saved artifact, generates predictions, and returns JSON for the dashboard to display.

## Important Modeling Assumptions

- The training history ends on **August 15, 2017**; the current forecast begins on August 16, 2017. Results are historical demonstrations, not current retail forecasts.
- Promotions must be provided explicitly or assumed to be zero. The current dashboard uses the zero-promotion assumption.
- Future lag and rolling features use recursively generated predictions rather than unavailable actual future sales.
- Predicted sales are non-negative and represent units, not revenue.

## Roadmap

1. Complete and validate the forecasting API and dashboard.
2. Deploy the frontend and backend and test the live application.
3. Add a deterministic what-if scenario planner with baseline comparisons.
4. Explore a natural-language interface that translates user requests into validated scenario inputs.

## Project Status

The local forecasting engine, FastAPI backend, and initial dashboard have been implemented. The production frontend build succeeds. Public deployment and scenario planning remain in progress.

*This is a learning and portfolio project focused on applied machine learning, API development, frontend engineering, and end-to-end deployment.*


## Workflow



## Technology	What it does in DemandLens
JavaScript	The programming language that runs in the browser and in Node.js.
TypeScript	JavaScript with a type system. It catches many mistakes before the code runs.
React	Builds the interface from reusable components and updates it when data changes.
Next.js	A framework built around React that provides routing, development tools, and production builds.
Node.js	Runs JavaScript outside the browser. Your Next.js development and build tools use it.
Recharts	A React library that renders charts from data.
FastAPI	Your Python API server, which calls the forecasting model.


2. Important engineering decisions to remember
These are the distinctions worth keeping in your personal notes as you build the project.
Decision	Why it matters
LightGBM stays in Python	The browser does not need to run the ML model. FastAPI handles inference.
React + TypeScript + Next.js	React builds the UI; TypeScript adds static type checking; Next.js provides the app framework and production build tools.
Frontend and backend are separate services	They can be deployed independently, but the frontend must know the API URL and the backend must allow the frontend's origin through CORS.
Development versus production	npm run dev is for local development. npm run build verifies a production build. npm run start serves that build locally.
Recursive forecasting	Predictions for earlier future dates feed into lag and rolling features for later dates. Errors can accumulate over the forecast horizon.
Promotion assumptions	Missing future promotion data is not the same as zero promotions. The app should make assumptions explicit.
Historical versus live predictions	Hosting a model doesn't refresh its training data. DemandLens currently forecasts from 2017 historical data.
Git versus GitHub	Git tracks project versions locally; GitHub hosts the remote repository. Hosting platforms can deploy from the remote repository.