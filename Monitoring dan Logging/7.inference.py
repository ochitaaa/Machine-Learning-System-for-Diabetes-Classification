import requests

# Contoh data input (sesuaikan dengan dataset Anda)
data = {
    "inputs": [[6, 148, 72, 35, 0, 33.6, 0.627, 50, 1, 60, 0.5, 37, 44, 121, 21, 89, 0.11]]}


# Kirim ke model yang sedang di-servis
response = requests.post(
    "http://localhost:8080/invocations",
    json=data,
    headers={"Content-Type": "application/json"}
)

# Tampilkan hasil
print("Status Code:", response.status_code)
print("Prediksi:", response.json())