import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def send_credentials_email(email, emp_id, temp_password):
    try:
        # Create a multipart message for better email formatting
        msg = MIMEMultipart()
        msg['Subject'] = 'Your New Account Credentials'
        msg['From'] = os.getenv('EMAIL_FROM')
        msg['To'] = email
        
        # Email body with better formatting
        body = f"""
        Hello,
        
        Your account has been created in our AI Task Management.
        
        Employee ID: {emp_id}
        Temporary Password: {temp_password}
        
        Please login and change your password immediately for security purposes.
        
        This is an automated message, please do not reply.
        """
        
        # Attach the body to the message
        msg.attach(MIMEText(body, 'plain'))
        
        # Gmail SMTP settings
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        
        # For Gmail, you need to use your Gmail address and app password
        email_user = os.getenv('EMAIL_USER')
        email_password = os.getenv('EMAIL_PASSWORD')
        
        # Connect to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()  # Identify to SMTP server
            server.starttls()  # Secure the connection
            server.ehlo()  # Re-identify over TLS connection
            server.login(email_user, email_password)
            server.send_message(msg)
            print(f"Email sent successfully to {email}")
            return True
            
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False