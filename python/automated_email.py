import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SENDER = os.environ.get("SENDER_EMAIL")
PASSWORD = os.environ.get("EMAIL_PASSWORD")


class AutomatedEmail:
    def __init__(self):
        self.message = MIMEMultipart("alternative")
        self.html_str = ""

    def attach_html(self):
        self.message.attach(MIMEText(self.html_str, _subtype="html"))

    def attach_csv(self, csv_dir: str = "../temp"):
        for file in os.scandir(csv_dir):
            with open(file.path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={file.name}")
                self.message.attach(part)

    def send_email(self):
        self.attach_html()
        self.attach_csv()
        with smtplib.SMTP(host="smtp.gmail.com", port=587, timeout=10) as connection:
            connection.starttls()
            connection.login(user=SENDER, password=PASSWORD)
            connection.send_message(msg=self.message)
