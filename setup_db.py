#!/usr/bin/env python3
# setup_db.py
import string
import random
import sys

try:
    from employee_service import app, db, Employee
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure your employee_service module is properly installed or in your PYTHONPATH")
    sys.exit(1)

# Define generate_random_password if it's not available from the module
def generate_random_password(length=5):
    """Generate a random password with the specified length"""
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(length))

try:
    with app.app_context():
        # Drop all tables and recreate them with CASCADE
        # Using raw SQL with text() to ensure CASCADE is used
        from sqlalchemy import text
        
        db.session.execute(text("DROP SCHEMA public CASCADE"))
        db.session.execute(text("CREATE SCHEMA public"))
        db.session.execute(text("GRANT ALL ON SCHEMA public TO postgres"))
        db.session.execute(text("GRANT ALL ON SCHEMA public TO public"))
        db.session.commit()
        
        # Now create all tables from scratch
        db.create_all()
        print("Tables dropped with CASCADE and recreated!")

        # Try to import generate_random_password from the module
        try:
            from employee_service import generate_random_password
        except ImportError:
            # Use our local implementation if not available in the module
            print("Using local implementation of generate_random_password")
        
        # Create admin user
        temp_password = generate_random_password()
        admin = Employee(
            emp_id='admin', 
            name='Administrator',
            email='admin@company.com',
            role='admin'
        )
        
        # Check if set_password method exists
        if hasattr(admin, 'set_password'):
            admin.set_password(temp_password)
        else:
            # Fallback if set_password doesn't exist
            print("Warning: Employee model has no set_password method")
            # Try common alternatives
            if hasattr(admin, 'password'):
                admin.password = temp_password
            else:
                print("Error: Could not set password on Employee model")
                sys.exit(1)
        
        db.session.add(admin)
        db.session.commit()
        print(f"Admin created with temporary password: {temp_password}")
        print("Database setup completed successfully!")

except Exception as e:
    print(f"Error during database setup: {e}")
    sys.exit(1)