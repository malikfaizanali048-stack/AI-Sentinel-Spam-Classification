import os
from dotenv import load_dotenv
from imap_tools import MailBox, AND

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

def fetch_new_emails_since(last_uid=None, initial_limit=10):
    emails = []
    print("Connecting to mailbox...", flush=True)
    with MailBox('imap.gmail.com').login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD) as mailbox:
        print("Logged in, building criteria...", flush=True)
        if last_uid:
            criteria = AND(uid=f'{int(last_uid)+1}:*')
            msgs = mailbox.fetch(criteria)
        else:
            # first run: just get the most recent N emails, not the whole mailbox
            msgs = mailbox.fetch(AND(all=True), limit=initial_limit, reverse=True)

        print("Fetch call returned, iterating messages...", flush=True)
        for msg in msgs:
            email_data = {
                "uid": msg.uid,
                "subject": msg.subject,
                "from": msg.from_,
                "date": str(msg.date),
                "body": msg.text or msg.html or "",
                "headers": dict(msg.headers),
            }
            emails.append(email_data)
    print(f"Fetched {len(emails)} emails this cycle", flush=True)
    return emails

# Alias / Wrapper function for app.py
def fetch_recent_emails(limit=10):
    return fetch_new_emails_since(last_uid=None, initial_limit=limit)