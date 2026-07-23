from flask import Flask, jsonify, request, render_template
import sys, json, email, os, uuid
from email import policy
from datetime import datetime
from urllib.parse import quote
sys.path.append("src")

import joblib
from models.embed_text import load_embedding_model
from models.score_anomaly import load_autoencoder
from ingestion.fetch import fetch_recent_emails
from pipeline.run_pipeline import process_email

app = Flask(__name__)

print("Loading models...")
classifier = joblib.load("data/processed/lgbm_model_enhanced.pkl")
embed_model = load_embedding_model()
autoencoder, scaler = load_autoencoder()
print("Models loaded.")

HISTORY_PATH = "data/processed/scan_history.json"


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def save_history_entry(entry):
    history = load_history()
    entry["id"] = str(uuid.uuid4())
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(entry)
    history = history[-200:]
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def extract_message_id(headers):
    """Case-insensitive lookup of the Message-ID header, handling tuple/list values from imap_tools."""
    headers = headers or {}
    for key, value in headers.items():
        if key.lower() == "message-id":
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            return str(value).strip("<>")
    return ""


@app.route("/")
def screen1():
    return render_template("screen1.html")


@app.route("/manual")
def screen2():
    return render_template("screen2.html")


@app.route("/api/check-inbox")
def check_inbox():
    emails = fetch_recent_emails(limit=10)  # <-- CHANGE THIS NUMBER to adjust inbox check limit
    results = []

    for e in emails:
        label, prob = process_email(e, classifier, embed_model, autoencoder, scaler)

        msg_id = extract_message_id(e.get("headers", {}))
        print(f"DEBUG - Subject: {e.get('subject')} | Message-ID extracted: '{msg_id}'")

        gmail_link = None
        if msg_id:
            gmail_link = f"https://mail.google.com/mail/u/0/#search/rfc822msgid:{quote(msg_id)}"

        results.append({
            "subject": e.get("subject"),
            "from": e.get("from"),
            "label": label,
            "score": round(float(prob), 3),
            "link": gmail_link
        })

    return jsonify(results)


@app.route("/api/check-single", methods=["POST"])
def check_single():
    data = request.json
    subject = data.get("subject", "")
    body = data.get("body", "")

    email_data = {"subject": subject, "body": body, "from": "", "headers": {}}
    label, prob = process_email(email_data, classifier, embed_model, autoencoder, scaler)
    result = {"label": label, "score": round(float(prob), 3)}

    save_history_entry({
        "type": "manual",
        "subject": subject,
        "body": body,
        "label": label,
        "score": result["score"]
    })

    return jsonify(result)


@app.route("/api/check-batch-files", methods=["POST"])
def check_batch_files():
    files = request.files.getlist("files")
    results = []

    for f in files[:50]:
        try:
            if f.filename.endswith(".json"):
                data = json.load(f)
                subject, body = data.get("subject", ""), data.get("body", "")
            else:
                msg = email.message_from_bytes(f.read(), policy=policy.default)
                subject = msg['subject'] or ""
                if msg.is_multipart():
                    body = next(
                        (p.get_content() for p in msg.walk() if p.get_content_type() == "text/plain"),
                        ""
                    )
                else:
                    body = msg.get_content()

            email_data = {"subject": subject, "body": body, "from": "", "headers": {}}
            label, prob = process_email(email_data, classifier, embed_model, autoencoder, scaler)
            result = {"filename": f.filename, "subject": subject, "label": label, "score": round(float(prob), 3)}
            results.append(result)

            save_history_entry({
                "type": "batch",
                "filename": f.filename,
                "subject": subject,
                "body": body,
                "label": label,
                "score": result["score"]
            })
        except Exception as ex:
            results.append({"filename": f.filename, "error": str(ex)})

    return jsonify(results)


@app.route("/api/history")
def get_history():
    return jsonify(list(reversed(load_history())))


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    with open(HISTORY_PATH, "w") as f:
        json.dump([], f)
    return jsonify({"status": "cleared"})


@app.route("/api/history/delete/<entry_id>", methods=["POST"])
def delete_history_entry(entry_id):
    history = load_history()
    history = [h for h in history if h.get("id") != entry_id]
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)