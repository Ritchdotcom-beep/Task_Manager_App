from email_services import send_credentials_email

@app.route('/test_email')
def test_email():
    test_email = "recipient@example.com"  # Use a real email you can check
    result = send_credentials_email(
        email=test_email,
        emp_id="TEST123",
        temp_password="testpass123"
    )
    return jsonify({"success": result})