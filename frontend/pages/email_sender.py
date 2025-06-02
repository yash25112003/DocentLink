import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd

# Enter your Email account credentials
your_email = "yashshah25112003@gmail.com"
# IMPORTANT: Replace "your_actual_app_password" with the 16-character App Password
# you generate from your Google Account settings. Do not use your regular Gmail password here.
your_password = "emarfcfovipitwzp" # <-- REPLACE THIS

# Email content
subject = "Exploring Research Internship Opportunity in field of Computer Science and Engineering"
body = (
    "Respected Professor,\n\n"
    "I am Yash Shah, a Computer Engineering student ........"  # Fill in your details
    # ... (ensure the rest of your email body is complete here) ...
    "I am highly motivated to contribute to cutting-edge research at your esteemed institution and gain further hands-on experience in AI/ML and IoT. I have attached the link to my resume for your consideration and would be grateful for the opportunity to discuss how I can contribute to your research endeavors.\n\n"
    "Please find my resume linked here: https://drive.google.com/view?usp=sharing\n\n" # Ensure this link works
    "Best regards,\n"
    "Yash Shah\n"
    "D.J. Sanghvi CoE\n"
    "Mumbai, Maharashtra, India\n"
    "+91 __________\n"  # Fill in your phone number
    "yashshah251103@gmail.com\n"
)

# Load professor emails
try:
    professors_df = pd.read_csv('TEST.csv') # Ensure this CSV file exists and has an 'email' column
    if 'email' not in professors_df.columns:
        print("Error: 'TEST.csv' must contain an 'email' column.")
        exit()
except FileNotFoundError:
    print("Error: 'TEST.csv' not found. Please make sure the file is in the same directory as the script.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: 'TEST.csv' is empty.")
    exit()


# Connect to Gmail's SMTP server
try:
    server = smtplib.SMTP(host='smtp.gmail.com', port=587)
    server.starttls() # Secure the connection
    server.login(your_email, your_password)
    print("Successfully logged into Gmail SMTP server.")
except smtplib.SMTPAuthenticationError as e:
    print(f"SMTP Authentication Error: {e}")
    print("Please ensure: ")
    print("1. Your email and App Password are correct.")
    print("2. You have generated an App Password from your Google Account (if 2-Step Verification is ON).")
    exit()
except Exception as e:
    print(f"An error occurred during SMTP server connection: {e}")
    exit()

# Loop through professors and send emails
for index, row in professors_df.iterrows():
    professor_email = row['email']
    if not isinstance(professor_email, str) or '@' not in professor_email:
        print(f"Skipping invalid email address at row {index}: {professor_email}")
        continue

    msg = MIMEMultipart()
    msg['From'] = your_email
    msg['To'] = professor_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server.send_message(msg)
        print(f"Email successfully sent to {professor_email}")
    except Exception as e:
        print(f"Failed to send email to {professor_email}. Error: {e}")
    del msg # Clean up message object

# Quit the server
server.quit()
print("Finished sending emails and disconnected from the server.")