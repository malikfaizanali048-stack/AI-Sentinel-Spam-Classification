import pandas as pd
import email
from email import policy

df = pd.read_csv("data/raw/spamassassin.csv")
raw_text = df.iloc[0]['text']

print("Raw text type:", type(raw_text))
print("Raw text preview:\n", raw_text[:300])
print("\n" + "="*50 + "\n")

try:
    msg = email.message_from_string(raw_text, policy=policy.default)
    print("Subject:", msg['subject'])
    print("From:", msg['from'])
    
    if msg.is_multipart():
        print("Message is multipart")
        for part in msg.walk():
            print("Part content type:", part.get_content_type())
    else:
        print("Message is NOT multipart")
        body = msg.get_content()
        print("Body preview:", body[:200])

except Exception as e:
    print("EXCEPTION during parsing:", repr(e))