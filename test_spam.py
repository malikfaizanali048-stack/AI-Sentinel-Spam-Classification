import sys
import os

# Ensure src is in Python path
sys.path.append("src")

from pipeline.run_pipeline import load_all_models, process_email

def test():
    print("Loading models via run_pipeline...")
    classifier, embed_model, autoencoder, scaler = load_all_models()

    # Obvious Spam Test Case
    test_email = {
        "subject": "URGENT: Your account has been suspended! Verify immediately",
        "body": "Dear customer, your bank account is locked. Click here http://phishing-site.com/login to enter your password and claim $10,000 USD prize.",
        "headers": {"Return-Path": "<security@fake-bank-alert.com>"},
        "from": "security@fake-bank-alert.com"
    }

    label, prob = process_email(test_email, classifier, embed_model, autoencoder, scaler)
    
    print("\n" + "="*30)
    print("--- TEST RESULT ---")
    print(f"Prediction Label : {label}")
    print(f"Spam Score/Prob  : {prob:.4f}")
    print("="*30)

if __name__ == "__main__":
    test()