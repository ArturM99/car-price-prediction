import datetime
import random
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import median

import psutil
import requests
from tqdm import tqdm

API_URL = "http://127.0.0.1:8002/predict"
ITERATIONS = 720  # number of data points (e.g. 12 hours, one per minute)


def load_cpu(duration: int) -> None:
    """
    Generates artificial CPU load for `duration` seconds.
    """
    start_time = time.time()
    while time.time() - start_time <= duration:
        _ = pow(random.randint(1, 1000), 32)


def generate_request_payload() -> dict:
    """
    Generates a random payload for a POST request.
    """
    return {
        "description": "Lorem ipsum dolor sit amet",
        "fuel": "gas",
        "id": random.randint(1, 2**32),
        "image_url": "https://images.craigslist.org/00808_i0faaALGPQxz_0CI0t2_600x450.jpg",
        "lat": 39.1618,
        "long": -76.6297,
        "manufacturer": "ford",
        "model": "mustang",
        "odometer": random.randint(1000, 500000),
        "posting_date": "2021-05-03T19:49:21-0400",
        "price": random.randint(1000, 500000),
        "region": "baltimore",
        "region_url": "https://baltimore.craigslist.org",
        "state": "md",
        "title_status": "clean",
        "transmission": "manual",
        "url": "https://baltimore.craigslist.org/cto/d/glen-burnie-mustang-50-convertible-2013/7316509996.html",
        "year": 2013,
    }


def write_sql_header(file) -> None:
    """
    Writes the database and table SQL schema.
    """
    header = """
    CREATE DATABASE IF NOT EXISTS metrics;
    USE metrics;

    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics (
        timestamp DATETIME NOT NULL,
        cpu_usage DECIMAL(5, 2) NOT NULL,
        mem_available BIGINT NOT NULL,
        reqs_per_min INT NOT NULL,
        time_of_proc DECIMAL(5, 2) NOT NULL,
        PRIMARY KEY (timestamp)
    ) ENGINE=InnoDB;

    INSERT INTO metrics (timestamp, cpu_usage, mem_available, reqs_per_min, time_of_proc) VALUES
    """
    file.write(header.strip() + "\n")


def collect_metrics(cpu_pool: ProcessPoolExecutor, current_time: datetime.datetime):
    """
    Over the course of 1 second:
    - sends requests
    - measures latency
    - counts the number of requests
    """
    start_time = time.time()
    latencies = []
    request_count = 0

    while True:
        request_start = time.time()
        try:
            requests.post(API_URL, json=generate_request_payload(), timeout=1)
        except requests.RequestException:
            # if a request fails — just skip it
            continue
        request_count += 1
        latencies.append(time.time() - request_start)

        if time.time() - start_time > 1:
            break

    # occasionally add extra CPU load
    if random.randint(0, 1):
        cpu_pool.submit(load_cpu, random.randint(1, 10))

    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().available
    latency_ms = median(latencies) * 1000 if latencies else 0

    return current_time, cpu, memory, request_count, latency_ms


def main(output_path: str) -> None:
    """
    Main function:
    - generates metrics
    - writes them to a SQL file
    """
    cpu_pool = ProcessPoolExecutor()
    current_time = datetime.datetime.utcnow() - datetime.timedelta(hours=10)

    with open(output_path, "w") as file:
        write_sql_header(file)

        for i in tqdm(range(ITERATIONS), desc="Generating metrics"):
            metrics = collect_metrics(cpu_pool, current_time)

            if i > 0:
                file.write(",\n")

            file.write(
                f'("{metrics[0]}", {metrics[1]}, {metrics[2]}, '
                f'{metrics[3]}, {metrics[4]:.2f})'
            )
            file.flush()

            current_time += datetime.timedelta(minutes=1)

        file.write(";\n")


if __name__ == "__main__":
    main("grafana.sql")
