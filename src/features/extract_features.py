import re
import pandas as pd
from urllib.parse import urlparse

def extract_url_features(body):
    body = str(body) if pd.notna(body) else ""
    urls = re.findall(r'https?://[^\s)>\]"]+', body)
    num_urls = len(urls)
    num_ip_urls = 0
    suspicious_tld = 0
    avg_url_length = 0
    suspicious_tlds = ['.xyz', '.top', '.click', '.link', '.zip', '.review']

    if urls:
        avg_url_length = sum(len(u) for u in urls) / len(urls)
        for u in urls:
            try:
                domain = urlparse(u).netloc
            except ValueError:
                continue  # skip malformed URLs (e.g., bad IPv6-like syntax)
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
                num_ip_urls += 1
            if any(domain.endswith(tld) for tld in suspicious_tlds):
                suspicious_tld += 1

    return {
        "num_urls": num_urls,
        "num_ip_urls": num_ip_urls,
        "suspicious_tld_count": suspicious_tld,
        "avg_url_length": avg_url_length
    }

def extract_header_features(headers, from_field):
    headers = headers or {}
    from_field = str(from_field) if pd.notna(from_field) else ""

    auth_results = headers.get("Authentication-Results", "") or ""
    auth_results = str(auth_results).lower()

    spf_pass = 1 if "spf=pass" in auth_results else 0
    dkim_pass = 1 if "dkim=pass" in auth_results else 0
    dmarc_pass = 1 if "dmarc=pass" in auth_results else 0

    reply_to = headers.get("Reply-To", "") or ""
    reply_to = str(reply_to) if pd.notna(reply_to) else ""

    from_domain = from_field.split("@")[-1].strip(">") if "@" in from_field else ""
    reply_domain = reply_to.split("@")[-1].strip(">") if "@" in reply_to else ""
    from_reply_mismatch = 1 if (reply_domain and from_domain and reply_domain != from_domain) else 0

    return {
        "spf_pass": spf_pass,
        "dkim_pass": dkim_pass,
        "dmarc_pass": dmarc_pass,
        "from_reply_mismatch": from_reply_mismatch
    }

def extract_text_signal_features(subject, body):
    subject = str(subject) if pd.notna(subject) else ""
    body = str(body) if pd.notna(body) else ""

    text = f"{subject} {body}".lower()
    urgency_words = ["urgent", "verify", "act now", "suspended", "click here", "limited time", "password"]
    urgency_count = sum(text.count(w) for w in urgency_words)
    exclaim_count = text.count("!")
    caps_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)

    return {
        "urgency_word_count": urgency_count,
        "exclamation_count": exclaim_count,
        "subject_caps_ratio": caps_ratio
    }

def extract_all_features(row):
    features = {}
    features.update(extract_url_features(row.get('body', "")))
    features.update(extract_header_features(row.get('headers', {}), row.get('from', "")))
    features.update(extract_text_signal_features(row.get('subject', ""), row.get('body', "")))
    return features