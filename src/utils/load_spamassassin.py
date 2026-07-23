import os
import glob
import pandas as pd
import email
from email import policy

def find_email_files(folder_path):
    """Recursively find all files, skip known non-email files like 'cmds'."""
    all_files = glob.glob(os.path.join(folder_path, "**", "*"), recursive=True)
    return [f for f in all_files if os.path.isfile(f) and os.path.basename(f) != "cmds"]

def load_folder(folder_path, label):
    data = []
    files = find_email_files(folder_path)
    print(f"  Found {len(files)} files in {folder_path}")
    
    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
                subject = msg['subject'] or ""
                sender = msg['from'] or ""
                headers = dict(msg.items())

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body = part.get_content()
                            except Exception:
                                body = ""
                            break
                else:
                    try:
                        body = msg.get_content()
                    except Exception:
                        body = ""

                data.append({
                    "subject": subject,
                    "from": sender,
                    "body": body,
                    "headers": headers,
                    "label": label
                })
        except Exception:
            continue
    return data

def load_spamassassin(base_path="data/raw/spamassassin"):
    all_data = []
    print("Loading spam...")
    all_data += load_folder(os.path.join(base_path, "spam"), 1)
    print("Loading easy_ham...")
    all_data += load_folder(os.path.join(base_path, "easy_ham"), 0)
    print("Loading hard_ham...")
    all_data += load_folder(os.path.join(base_path, "hard_ham"), 0)
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df = load_spamassassin()
    print("\nFinal shape:", df.shape)
    print(df['label'].value_counts())
    print("\nSample spam subject:", df[df['label']==1].iloc[0]['subject'])
    print("Sample spam body preview:", df[df['label']==1].iloc[0]['body'][:150])
    print("\nSample ham subject:", df[df['label']==0].iloc[0]['subject'])
    print("Sample ham body preview:", df[df['label']==0].iloc[0]['body'][:150])