import datetime
import random
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import median

import psutil
import requests
from tqdm import tqdm


API_URL = "http://127.0.0.1:8002/predict"
ITERATIONS = 720  # количество точек (например, 12 часов по минутам)


def load_cpu(duration: int) -> None:
    """
    Создаёт искусственную нагрузку на CPU в течение duration секунд.
    """
    start_time = time.time()

    while time.time() - start_time <= duration:
        _ = pow(random.randint(1, 1000), 32)


def generate_request_payload() -> dict:
    """
    Генерирует случайный payload для POST-запроса.
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
    Записывает SQL-структуру базы и таблицы.
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
    В течение 1 секунды:
    - отправляет запросы
    - считает latency
    - считает количество запросов
    """

    start_time = time.time()
    latencies = []
    request_count = 0

    while True:
        request_start = time.time()

        try:
            requests.post(API_URL, json=generate_request_payload(), timeout=1)
        except requests.RequestException:
            # если запрос упал — просто игнорируем
            continue

        request_count += 1
        latencies.append(time.time() - request_start)

        if time.time() - start_time > 1:
            break

    # иногда создаём дополнительную нагрузку
    if random.randint(0, 1):
        cpu_pool.submit(load_cpu, random.randint(1, 10))

    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().available
    latency_ms = median(latencies) * 1000 if latencies else 0

    return current_time, cpu, memory, request_count, latency_ms


def main(output_path: str) -> None:
    """
    Основная функция:
    - генерирует метрики
    - записывает их в SQL-файл
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