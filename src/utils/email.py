
import smtplib
import traceback as Traceback
import os
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

RECIPIENT = "thomas.bowman.2003@gmail.com"
SUBJECT = "AI training error encountered"

def ReportException(exception: Exception):
    load_dotenv()
    sender = os.getenv("EXCEPTION_EMAIL")
    if sender is None:
        raise Exception("Could not get email from .env file")
    sender_password = os.getenv("EXCEPTION_EMAIL_PASSWORD")
    if sender_password is None:
        raise Exception("Could not get email password from .env file")

    traceback = Traceback.format_exc()
    body = f"Exception encountered:\n{traceback}"

    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = RECIPIENT
    message["Subject"] = SUBJECT
    message.attach(MIMEText(body))

    smtp = smtplib.SMTP("smtp.gmail.com", 587)
    smtp.starttls()
    smtp.login(user=sender, password=sender_password)
    smtp.sendmail(sender, RECIPIENT, message.as_string())
    smtp.quit()

