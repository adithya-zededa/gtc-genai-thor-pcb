#!/usr/bin/env python3
"""
Gmail Configuration Helper for ZEDEDA Camera Agent
Helps set up Gmail authentication for email alerts
"""

import getpass
import smtplib
from email.message import EmailMessage

def test_gmail_auth(username, password, recipient):
    """Test Gmail authentication and send test email"""
    try:
        print(f"Testing Gmail authentication for: {username}")
        
        # Create test message
        msg = EmailMessage()
        msg['Subject'] = '🔧 ZEDEDA Camera Agent - Email Configuration Test'
        msg['From'] = username
        msg['To'] = recipient
        msg.set_content("""
ZEDEDA Camera Agent - Email Configuration Test

This is a test email to verify that the camera monitoring agent can send alerts.

If you received this email, the email configuration is working correctly!

The camera agent will now send alerts to this email address when computer monitors are detected.

Best regards,
ZEDEDA Camera Monitoring System
""")
        
        # Send via Gmail SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        
        print(f"✅ Test email sent successfully!")
        print(f"✅ From: {username}")
        print(f"✅ To: {recipient}")
        print("✅ Email configuration is working correctly!")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("\n🔧 Troubleshooting Steps:")
        print("1. Enable 2-Factor Authentication on your Gmail account")
        print("2. Generate an App Password:")
        print("   - Go to: https://myaccount.google.com/apppasswords")
        print("   - Select 'Mail' and your device")
        print("   - Use the generated 16-character password instead")
        print("3. Or enable 'Less secure app access' (not recommended):")
        print("   - Go to: https://myaccount.google.com/lesssecureapps")
        return False
        
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        return False

def main():
    print("🔧 ZEDEDA Camera Agent - Gmail Configuration Helper")
    print("=" * 60)
    
    # Get credentials
    username = input("Gmail username (ace7esstest@gmail.com): ").strip() or "ace7esstest@gmail.com"
    password = getpass.getpass("Gmail password/app password: ").strip()
    recipient = input("Recipient email (adithya7shankar@gmail.com): ").strip() or "adithya7shankar@gmail.com"
    
    print(f"\n📧 Testing email configuration...")
    success = test_gmail_auth(username, password, recipient)
    
    if success:
        print(f"\n✅ Email configuration successful!")
        print(f"The camera agent is ready to send alerts to {recipient}")
        
        # Update .env file
        env_content = f"""# ZEDEDA Camera Agent Environment Configuration
OLLAMA_URL=http://localhost:11434
VISION_MODEL=llava:7b
CAMERA_INDEX=0
CAPTURE_INTERVAL=5
LOG_LEVEL=INFO

# Email configuration
EMAIL_USER={username}
EMAIL_PASS={password}
EMAIL_FROM={username}

# SMTP settings
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=true
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Updated .env file with working credentials")
        
    else:
        print(f"\n❌ Email configuration failed!")
        print("Please fix the authentication issues and try again.")

if __name__ == "__main__":
    main()