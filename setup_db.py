#!/usr/bin/env python3
# setup_db.py
import string
import random
import sys

# First try to import both services
try:
    from employee_service import app as employee_app, db as employee_db, Employee
    has_employee_service = True
    print("Employee service module found.")
except ImportError as e:
    print(f"Employee service import error: {e}")
    print("Employee database setup will be skipped.")
    has_employee_service = False

try:
    from task_assignment_service import app as task_app, db as task_db, Task
    has_task_service = True
    print("Task assignment service module found.")
except ImportError as e:
    print(f"Task service import error: {e}")
    print("Task database setup will be skipped.")
    has_task_service = False

if not has_employee_service and not has_task_service:
    print("Neither employee_service nor task_assignment_service modules were found.")
    print("Make sure at least one of these modules is properly installed or in your PYTHONPATH.")
    sys.exit(1)

# Define generate_random_password if it's not available from the module
def generate_random_password(length=5):
    """Generate a random password with the specified length"""
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(length))

# Since they share the same database, only drop schema once
try:
    # Choose which app context to use - preferring employee_app if available
    app = employee_app if has_employee_service else task_app
    db = employee_db if has_employee_service else task_db
    
    with app.app_context():
        # Drop all tables and recreate them with CASCADE - DO THIS ONLY ONCE
        from sqlalchemy import text
        
        db.session.execute(text("DROP SCHEMA public CASCADE"))
        db.session.execute(text("CREATE SCHEMA public"))
        db.session.execute(text("GRANT ALL ON SCHEMA public TO postgres"))
        db.session.execute(text("GRANT ALL ON SCHEMA public TO public"))
        db.session.commit()
        print("Database schema reset completed!")
        
        # Create all tables from both services
        if has_employee_service:
            with employee_app.app_context():
                employee_db.create_all()
                print("Employee tables created!")
                
                # Create admin user
                try:
                    from employee_service import generate_random_password
                except ImportError:
                    print("Using local implementation of generate_random_password")
                
                temp_password = generate_random_password()
                admin = Employee(
                    emp_id='admin', 
                    name='Administrator',
                    email='admin@company.com',
                    role='admin'
                )
                
                if hasattr(admin, 'set_password'):
                    admin.set_password(temp_password)
                else:
                    if hasattr(admin, 'password'):
                        admin.password = temp_password
                    else:
                        print("Error: Could not set password on Employee model")
                
                employee_db.session.add(admin)
                employee_db.session.commit()
                print(f"Admin created with temporary password: {temp_password}")
        
        if has_task_service:
            with task_app.app_context():
                task_db.create_all()
                print("Task tables created!")
                
                # You can add sample tasks here if needed
                # Example:
                # sample_task = Task(
                #     task_id='TASK1001',
                #     project_type='website_development',
                #     skills=['HTML', 'CSS', 'JavaScript'],
                #     complexity='Medium',
                #     priority='High',
                #     status='unassigned'
                # )
                # task_db.session.add(sample_task)
                # task_db.session.commit()
                # print(f"Sample task created with ID: {sample_task.task_id}")
        
        print("Database setup completed successfully!")

except Exception as e:
    print(f"Error during database setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)