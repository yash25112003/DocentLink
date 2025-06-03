import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get email content from environment variables
subject = os.getenv('EMAIL_SUBJECT')
body = os.getenv('EMAIL_BODY')
recipient_name = os.getenv('RECIPIENT_NAME')
recipient_email = os.getenv('RECIPIENT_EMAIL')
sender_email = os.getenv('SENDER_EMAIL')
sender_password = os.getenv('SENDER_PASSWORD')

if not all([subject, body, recipient_email, sender_email, sender_password]):
    print("Error: Missing required environment variables.")
    exit(1)

try:
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    # Connect to Gmail's SMTP server
    server = smtplib.SMTP(host='smtp.gmail.com', port=587)
    server.starttls()
    server.login(sender_email, sender_password)
    
    # Send email
    server.send_message(msg)
    print(f"Email successfully sent to {recipient_name} at {recipient_email}")
    exit(0)
    
except Exception as e:
    print(f"Failed to send email: {str(e)}", file=sys.stderr)
    exit(1)
finally:
    try:
        server.quit()
    except:
        pass