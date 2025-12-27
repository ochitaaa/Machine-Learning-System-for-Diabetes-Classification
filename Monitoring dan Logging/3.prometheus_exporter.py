from flask import Flask, Response
import random
import psutil

app = Flask(__name__)

# Variabel untuk mengontrol kondisi alert
ALERT_MODE = {
    "accuracy_low": True,      # Aktifkan alert accuracy rendah
    "latency_high": True,      # Aktifkan alert latency tinggi
    "errors_high": True        # Aktifkan alert error tinggi
}

@app.route('/metrics')
def metrics():
    # --- METRIK DASAR DENGAN KONDISI ALERT ---
    if ALERT_MODE["accuracy_low"]:
        accuracy = round(0.75 + random.uniform(-0.03, 0.02), 4)  # di bawah 0.8 → trigger alert
    else:
        accuracy = round(0.85 + random.uniform(-0.03, 0.02), 4)

    if ALERT_MODE["latency_high"]:
        latency_ms = 250 + random.randint(50, 100)  # di atas 200 → trigger alert
    else:
        latency_ms = 120 + random.randint(-30, 40)

    request_count = random.randint(100, 200)

    if ALERT_MODE["errors_high"]:
        errors = random.randint(10, 20)  # di atas 5 → trigger alert
    else:
        errors = random.randint(0, 4)

    # --- METRIK TAMBAHAN ---
    prediction_count = request_count
    cpu_usage = round(psutil.cpu_percent(interval=0.1), 2)
    memory_usage = round(psutil.virtual_memory().used / (1024 * 1024), 2)  # MB
    batch_size = random.randint(1, 10)
    response_time_sec = round(latency_ms / 1000.0, 3)
    feature_drift = round(random.uniform(0.1, 0.3), 4)  # drift tinggi

    # --- FORMAT PROMETHEUS ---
    output = f"""# HELP mlflow_model_accuracy Model accuracy
# TYPE mlflow_model_accuracy gauge
mlflow_model_accuracy {accuracy}

# HELP mlflow_model_latency_ms Model latency in ms
# TYPE mlflow_model_latency_ms gauge
mlflow_model_latency_ms {latency_ms}

# HELP mlflow_model_requests_total Total requests
# TYPE mlflow_model_requests_total counter
mlflow_model_requests_total {request_count}

# HELP mlflow_model_errors_total Total errors
# TYPE mlflow_model_errors_total counter
mlflow_model_errors_total {errors}

# HELP mlflow_model_prediction_count Total predictions served
# TYPE mlflow_model_prediction_count counter
mlflow_model_prediction_count {prediction_count}

# HELP mlflow_model_cpu_usage_percent CPU usage during inference
# TYPE mlflow_model_cpu_usage_percent gauge
mlflow_model_cpu_usage_percent {cpu_usage}

# HELP mlflow_model_memory_usage_mb Memory usage in MB
# TYPE mlflow_model_memory_usage_mb gauge
mlflow_model_memory_usage_mb {memory_usage}

# HELP mlflow_model_batch_size Average batch size per request
# TYPE mlflow_model_batch_size gauge
mlflow_model_batch_size {batch_size}

# HELP mlflow_model_response_time_seconds Response time per prediction
# TYPE mlflow_model_response_time_seconds gauge
mlflow_model_response_time_seconds {response_time_sec}

# HELP mlflow_model_feature_drift_score Simulated feature drift score
# TYPE mlflow_model_feature_drift_score gauge
mlflow_model_feature_drift_score {feature_drift}
"""
    return Response(output, mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)