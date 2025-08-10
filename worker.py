# worker.py
import os
import tempfile
import base64
import json
import subprocess
import sys
import redis
from rq import Worker, Queue, Connection
import joblib

redis_conn = redis.Redis()

def create_venv_and_install(req_text, venv_path):
    # Create venv
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    pip_path = os.path.join(venv_path, "bin", "pip")
    # Upgrade pip
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    if req_text:
        # Write requirements.txt
        req_file = os.path.join(venv_path, "requirements.txt")
        with open(req_file, "w") as f:
            f.write(req_text)
        # Install dependencies
        subprocess.run([pip_path, "install", "-r", req_file], check=True)

def run_prediction(venv_path, model_path, data_json):
    # Run a subprocess inside venv to execute prediction script
    python_path = os.path.join(venv_path, "bin", "python")
    script = f"""
import joblib
import json
model = joblib.load('{model_path}')
data = {data_json}
pred = model.predict([list(data.values())])
print(json.dumps(pred.tolist()))
"""
    result = subprocess.run([python_path, "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Prediction error: {result.stderr}")
    return json.loads(result.stdout)

def process_job(job_id, model_b64, data, requirements=None):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = os.path.join(tmpdir, "venv")
            create_venv_and_install(requirements, venv_path)
            # Save model
            model_path = os.path.join(tmpdir, "model.pkl")
            with open(model_path, "wb") as f:
                f.write(base64.b64decode(model_b64))
            # Run prediction
            prediction = run_prediction(venv_path, model_path, json.dumps(data))
            # Save result in Redis
            redis_conn.set(f"result:{job_id}", json.dumps(prediction))
    except Exception as e:
        redis_conn.set(f"result:{job_id}", json.dumps({"error": str(e)}))

