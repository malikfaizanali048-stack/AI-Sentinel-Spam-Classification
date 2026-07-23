import json

with open("data/raw/ephishllm/ephishLLM.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total entries:", len(data))
print("\nFirst entry keys:", data[0].keys())
print("\nFirst entry:")
print(data[0])
print("\nSecond entry:")
print(data[1])