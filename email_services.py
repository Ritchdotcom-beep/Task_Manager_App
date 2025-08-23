# Simplified email_services.py without company name dependency

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def send_email(to_email, subject, html_body, text_body=None):
    """Generic email sending function"""
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = os.getenv('EMAIL_USER')
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add text and HTML parts
        if text_body:
            text_part = MIMEText(text_body, 'plain')
            msg.attach(text_part)
        
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)

        # Gmail SMTP settings
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        email_user = os.getenv('EMAIL_USER')
        email_password = os.getenv('EMAIL_PASSWORD')

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(email_user, email_password)
            server.send_message(msg)
        
        print(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send email to {to_email}: {str(e)}")
        return False

def send_credentials_email(email, emp_id, temp_password):
    """Send credentials email to new employee"""
    subject = 'Your New Account Credentials'
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
            .container {{ background-color: white; padding: 30px; border-radius: 8px; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ color: #007bff; border-bottom: 2px solid #007bff; padding-bottom: 15px; margin-bottom: 25px; }}
            .content {{ line-height: 1.6; color: #333; }}
            .credentials {{ background-color: #e7f3ff; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff; }}
            .warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #ffc107; }}
            .button {{ background-color: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 15px 0; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>🎉 Welcome to AI Task Management System</h2>
            </div>
            <div class="content">
                <p>Hello,</p>
                
                <p>Your account has been created in our AI Task Management System. Welcome aboard!</p>
                
                <div class="credentials">
                    <h3>🔐 Your Login Credentials:</h3>
                    <strong>Employee ID:</strong> {emp_id}<br>
                    <strong>Temporary Password:</strong> {temp_password}
                </div>
                
                <div class="warning">
                    <strong>⚠️ Important Security Notice:</strong><br>
                    Please login and change your password immediately for security purposes.
                </div>
                
                <p><strong>Getting Started:</strong></p>
                <ol>
                    <li>Visit the employee portal</li>
                    <li>Select your assigned role</li>
                    <li>Log in with your credentials above</li>
                    <li>Change your password when prompted</li>
                    <li>Complete your profile setup</li>
                </ol>
                
                <p>If you have any questions or need assistance, please don't hesitate to contact the system administrator.</p>
                
                <p>Best regards,<br>
                AI Task Management System</p>
            </div>
            <div class="footer">
                <p>This email contains sensitive login information. Please keep it secure and delete after changing your password.</p>
                <p>This is an automated message, please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Welcome to AI Task Management System
    
    Hello,
    
    Your account has been created in our AI Task Management System.
    
    Your Login Credentials:
    Employee ID: {emp_id}
    Temporary Password: {temp_password}
    
    IMPORTANT: Please login and change your password immediately for security purposes.
    
    Getting Started:
    1. Visit the employee portal
    2. Select your assigned role
    3. Log in with your credentials above
    4. Change your password when prompted
    5. Complete your profile setup
    
    If you have any questions, please contact the system administrator.
    
    Best regards,
    AI Task Management System
    
    This is an automated message, please do not reply.
    """
    
    return send_email(email, subject, html_body, text_body)

def send_new_application_notification(admin_email, applicant_name, applicant_email, requested_role):
    """Notify admin of new employee application"""
    subject = f"New Employee Application - {applicant_name}"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
            .container {{ background-color: white; padding: 30px; border-radius: 8px; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 15px; margin-bottom: 25px; }}
            .content {{ line-height: 1.6; color: #333; }}
            .highlight {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .button {{ background-color: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 5px; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>🔔 New Employee Application Pending Review</h2>
            </div>
            <div class="content">
                <p>Dear Admin,</p>
                
                <p>A new employee application has been submitted and requires your review and approval.</p>
                
                <div class="highlight">
                    <strong>Application Details:</strong><br>
                    <strong>Name:</strong> {applicant_name}<br>
                    <strong>Email:</strong> {applicant_email}<br>
                    <strong>Requested Role:</strong> {requested_role}<br>
                    <strong>Submitted:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </div>
                
                <p><strong>Next Steps:</strong></p>
                <ul>
                    <li>Log into the admin dashboard to review the full application</li>
                    <li>Review the applicant's qualifications and experience</li>
                    <li>Approve and assign the final system role, or reject with feedback</li>
                </ul>
                
                <p>Please review this application promptly to ensure a smooth onboarding process.</p>
                
                <p>Best regards,<br>
                AI Task Management HR System</p>
            </div>
            <div class="footer">
                <p>This is an automated notification from the AI Task Management Employee Management System.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    New Employee Application Pending Review
    
    Dear Admin,
    
    A new employee application has been submitted and requires your review.
    
    Application Details:
    Name: {applicant_name}
    Email: {applicant_email}
    Requested Role: {requested_role}
    Submitted: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
    
    Please log into the admin dashboard to review and process this application.
    
    Best regards,
    AI Task Management HR System
    """
    
    return send_email(admin_email, subject, html_body, text_body)

def send_approval_notification(email, name, emp_id, role, temp_password):
    """Send approval notification with login credentials"""
    subject = "Welcome to AI Task Management - Application Approved!"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
            .container {{ background-color: white; padding: 30px; border-radius: 8px; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ color: #28a745; border-bottom: 2px solid #28a745; padding-bottom: 15px; margin-bottom: 25px; }}
            .content {{ line-height: 1.6; color: #333; }}
            .credentials {{ background-color: #d4edda; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #28a745; }}
            .warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #ffc107; }}
            .button {{ background-color: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 15px 0; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>🎉 Congratulations! Your Application Has Been Approved</h2>
            </div>
            <div class="content">
                <p>Dear {name},</p>
                
                <p>Great news! Your application to join our AI Task Management System has been approved. Welcome to the team!</p>
                
                <div class="credentials">
                    <h3>🔐 Your Login Credentials:</h3>
                    <strong>Employee ID:</strong> {emp_id}<br>
                    <strong>Role:</strong> {role.title()}<br>
                    <strong>Temporary Password:</strong> {temp_password}
                </div>
                
                <div class="warning">
                    <strong>⚠️ Important Security Notice:</strong><br>
                    You will be required to change your password on your first login for security purposes.
                </div>
                
                <p><strong>Getting Started:</strong></p>
                <ol>
                    <li>Access the employee portal</li>
                    <li>Select your role: <strong>{role.title()}</strong></li>
                    <li>Log in with your credentials above</li>
                    <li>Change your password when prompted</li>
                    <li>Complete your profile setup</li>
                </ol>
                
                <p>If you have any questions or need assistance, please don't hesitate to contact the HR department.</p>
                
                <p>We look forward to working with you!</p>
                
                <p>Best regards,<br>
                AI Task Management Administration Team</p>
            </div>
            <div class="footer">
                <p>This email contains sensitive login information. Please keep it secure and delete after changing your password.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Congratulations! Your Application Has Been Approved
    
    Dear {name},
    
    Your application to join our AI Task Management System has been approved. Welcome to the team!
    
    Your Login Credentials:
    Employee ID: {emp_id}
    Role: {role.title()}
    Temporary Password: {temp_password}
    
    IMPORTANT: You will be required to change your password on your first login.
    
    Getting Started:
    1. Access the employee portal
    2. Select your role: {role.title()}
    3. Log in with your credentials above
    4. Change your password when prompted
    5. Complete your profile setup
    
    Welcome to AI Task Management!
    
    Best regards,
    Administration Team
    """
    
    return send_email(email, subject, html_body, text_body)

def send_rejection_notification(email, name, reason):
    """Send rejection notification"""
    subject = "Application Update - AI Task Management"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
            .container {{ background-color: white; padding: 30px; border-radius: 8px; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ color: #dc3545; border-bottom: 2px solid #dc3545; padding-bottom: 15px; margin-bottom: 25px; }}
            .content {{ line-height: 1.6; color: #333; }}
            .reason {{ background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #dc3545; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Application Status Update</h2>
            </div>
            <div class="content">
                <p>Dear {name},</p>
                
                <p>Thank you for your interest in joining our AI Task Management System. After careful review, we regret to inform you that we cannot move forward with your application at this time.</p>
                
                <div class="reason">
                    <strong>Feedback:</strong><br>
                    {reason}
                </div>
                
                <p>We appreciate the time you took to apply and encourage you to consider applying for future opportunities that match your qualifications.</p>
                
                <p>If you have any questions about this decision, please feel free to contact our HR department.</p>
                
                <p>Thank you for your understanding.</p>
                
                <p>Best regards,<br>
                AI Task Management HR Department</p>
            </div>
            <div class="footer">
                <p>This is an automated notification from the AI Task Management Employee Management System.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Application Status Update
    
    Dear {name},
    
    Thank you for your interest in joining our AI Task Management System. After careful review, we cannot move forward with your application at this time.
    
    Feedback: {reason}
    
    We appreciate your interest and encourage you to apply for future opportunities.
    
    Best regards,
    AI Task Management HR Department
    """
    
    return send_email(email, subject, html_body, text_body)