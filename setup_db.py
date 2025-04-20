# modified setup_db.py
from employee_service import app, db

with app.app_context():
    # Drop all tables and recreate them
    db.drop_all()
    db.create_all()
    print("Tables dropped and recreated!")

    # Create admin user
    from employee_service import Employee, generate_random_password
    temp_password = generate_random_password()
    admin = Employee(
        emp_id='admin', 
        name='Administrator',
        email='admin@company.com',
        role='admin'
    )
    admin.set_password(temp_password)
    db.session.add(admin)
    db.session.commit()
    print(f"Admin created with temporary password: {temp_password}")