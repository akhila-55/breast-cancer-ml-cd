import json

QUALITY_THRESHOLD = 0.90

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Required Accuracy: {QUALITY_THRESHOLD:.2f}")

if accuracy < QUALITY_THRESHOLD:
    raise ValueError(
        f"Quality Gate Failed: accuracy {accuracy:.4f} "
        f"is below {QUALITY_THRESHOLD:.2f}"
    )

print("Quality Gate Passed!")
