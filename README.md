# Car Price Category Prediction

The service predicts the **price category** of a used car (`price_category`) based on its characteristics — model, year, mileage, condition, etc. Beyond the ML model itself, the project includes infrastructure for load testing the service and real-time monitoring through Grafana.

## Project structure

```
.
├── main.py                 # FastAPI inference service
├── load.py                 # load generator + metrics collection (CPU, memory, latency)
├── schedule.py              # periodic batch inference on a schedule (APScheduler)
├── docker-compose.yml       # monitoring stack: MariaDB + Adminer + Grafana
├── init/
│   └── grafana.sql          # SQL dump with dashboard metrics
├── Model/
│   ├── pipeline.py          # training and best-model selection
│   ├── cars_pipe.pkl        # trained model (created by pipeline.py)
│   └── Data/
│       └── homework.csv     # source data (not included in the repo, see below)
└── requirements.txt
```

## Installation

```
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Minimal dependencies:

```
fastapi
uvicorn
pandas
scikit-learn
dill
requests
psutil
tqdm
apscheduler
tzlocal
```

## Data

The `Model/Data/homework.csv` file is not included in the repo (too large for git). Place a dataset of used-car listings in this folder before training the model. Expected columns: `id`, `url`, `region`, `region_url`, `price`, `year`, `manufacturer`, `model`, `fuel`, `odometer`, `transmission`, `title_status`, `image_url`, `description`, `state`, `lat`, `long`, `posting_date`, `price_category` (target variable).

## Training the model

```
python Model/pipeline.py
```

The script:

1. Reads `Model/Data/homework.csv`.
2. Prepares features: drops uninformative columns (`id`, `url`, `region`, `image_url`, `description`, etc.), clips outliers in the manufacture year (interquartile range method), adds derived features `short_model` (first word of the model name) and `age_category` (new/mid/old).
3. Encodes categorical features via `OneHotEncoder`, scales numerical ones via `StandardScaler`.
4. Compares three models — `LogisticRegression`, `RandomForestClassifier`, `SVC` — via 4-fold cross-validation and picks the best one by accuracy.
5. Trains the best model on the full dataset and saves it together with metadata to `cars_pipe.pkl` (via `dill`, to preserve custom preprocessing functions).

## Running the API

```
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

Swagger docs: `http://localhost:8002/docs`

### Endpoints

| Method | Path       | Description                                       |
| ------ | ---------- | -------------------------------------------------- |
| GET    | `/status`  | Service health check                                |
| GET    | `/version` | Model metadata (model type, accuracy, training date) |
| POST   | `/predict` | Predict price category for a listing                |

### Example `/predict` response

```
{
  "ID": 1234567890,
  "Pred": "medium",
  "Price": 15000
}
```

## Load testing and monitoring

The project includes a mini stack for observing the service's behavior under load:

- **`load.py`** — sends a batch of requests to `/predict` with random data once per second, occasionally generating artificial CPU load in parallel, and every second measures: CPU usage, available memory, number of processed requests, and median response latency. Results are written to a SQL file (`grafana.sql`), ready to be loaded into the database.
- **`docker-compose.yml`** — spins up:
  * `MariaDB` — storage for the collected metrics;
  * `Adminer` — a web UI for browsing the database (`http://localhost:8080`);
  * `Grafana` — dashboards for visualizing the metrics (`http://localhost:3000`).
- **`schedule.py`** — demonstrates periodic batch inference: every 5 seconds it takes a random sample of rows from the dataset and runs them through the model (a useful example of the "scheduled prediction" production pattern, not just on-demand).

### Running the monitoring stack

```
docker-compose up -d
```
> ⚠️ Before the first run, change the default passwords in `docker-compose.yml` (`MARIADB_ROOT_PASSWORD`, `MARIADB_PASSWORD`) — the values in the repo are only placeholders.

Once the service (`main.py`) and containers are running, start the load generator:

```
python load.py
```

It will produce a `grafana.sql` file with the collected metrics, which can be imported into MariaDB via Adminer to build a Grafana dashboard.

## Notes

- `main.py` and `Model/pipeline.py` use the same preprocessing functions (`filter_data`, `year_outliers_clean`, `short_model`, `age_category`), so data handling is identical during training and inference.
- The target variable is a price category (classification), not the exact car price.
- The model is automatically selected from three candidates via cross-validation — the winning model's type is shown in the `/version` endpoint response.
-e 
---
🇷🇺 [Читать на русском](https://github.com/ArturM99/car-price-prediction/tree/RU)
