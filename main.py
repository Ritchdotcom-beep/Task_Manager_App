from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import string
import random
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

# Add the database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)
class Employee(db.Model):
    __tablename__ = 'employees'
    
    emp_id = db.Column(db.String(20), primary_key=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_first_login = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Employee {self.emp_id}>'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
# Function to generate random password
def generate_random_password(length=5):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))   



@app.route("/")
def index():
    return render_template("index.html")


# Route to select role and redirect to login
@app.route('/select_role/<role>')
def select_role(role):
    if role not in ['developer', 'project_manager', 'human_resource']:
        flash('Invalid role selected')
        return redirect(url_for('index'))
    
    session['selected_role'] = role
    return redirect(url_for('login'))

# Route for login with role-specific handling
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'selected_role' not in session:
        flash('Please select a role first')
        return redirect(url_for('index'))
    
    role = session['selected_role']
    
    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        password = request.form.get('password')
        
        employee = Employee.query.filter_by(emp_id=emp_id).first()
        
        if not employee:
            flash('Employee ID not found')
            return render_template('login.html', role=role)
        
        if employee.role != role.replace('_', ' '):
            flash(f'This employee ID is not registered as a {role.replace("_", " ")}')
            return render_template('login.html', role=role)
        
        if employee and employee.check_password(password):
            session['emp_id'] = emp_id
            session['role'] = employee.role
            
            # Update last login timestamp
            employee.last_login = datetime.utcnow()
            db.session.commit()
            
            # If first login, redirect to password change page
            if employee.is_first_login:
                return redirect(url_for('change_password'))
            
            # Otherwise redirect to appropriate dashboard based on role
            if role == 'developer':
                return redirect(url_for('developer_dashboard'))
            elif role == 'project_manager':
                return redirect(url_for('manager_dashboard'))
            else:
                return redirect(url_for('hr_dashboard'))
        else:
            flash('Invalid employee ID or password')
    
    return render_template('login.html', role=role)

# Route for changing password
@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'emp_id' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        employee = Employee.query.filter_by(emp_id=session['emp_id']).first()
        
        if not employee.check_password(current_password):
            flash('Current password is incorrect')
            return render_template('change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match')
            return render_template('change_password.html')
        
        # Update password and set first_login to False
        employee.set_password(new_password)
        employee.is_first_login = False
        db.session.commit()
        
        flash('Password changed successfully')
        
        # Redirect based on role
        role = session['role'].lower().replace(' ', '_')
        if role == 'developer':
            return redirect(url_for('developer_dashboard'))
        elif role == 'project_manager':
            return redirect(url_for('manager_dashboard'))
        else:
            return redirect(url_for('hr_dashboard'))
    
    return render_template('change_password.html', first_login=Employee.query.filter_by(emp_id=session['emp_id']).first().is_first_login)

# Role-specific dashboards
@app.route('/developer_dashboard')
def developer_dashboard():
    if 'emp_id' not in session or session['role'] != 'developer':
        return redirect(url_for('index'))
    return render_template('developer_dashboard.html')

@app.route('/manager_dashboard')
def manager_dashboard():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return redirect(url_for('index'))
    return render_template('manager_dashboard.html')

@app.route('/hr_dashboard')
def hr_dashboard():
    if 'emp_id' not in session or session['role'] != 'human resource':
        return redirect(url_for('index'))
    return render_template('hr_dashboard.html')

# Admin route to create a new employee
@app.route('/admin/create_employee', methods=['GET', 'POST'])
def create_employee():
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        role = request.form.get('role')
        
        # Check if employee already exists
        existing_employee = Employee.query.filter_by(emp_id=emp_id).first()
        if existing_employee:
            flash('Employee ID already exists')
            return render_template('create_employee.html')
        
        # Generate random password
        temp_password = generate_random_password()
        
        # Create new employee
        new_employee = Employee(emp_id=emp_id, role=role)
        new_employee.set_password(temp_password)
        
        db.session.add(new_employee)
        db.session.commit()
        
        flash(f'Employee created with ID: {emp_id} and temporary password: {temp_password}')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('create_employee.html')

# Admin dashboard
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'emp_id' not in session or session['role'] != 'admin':
        return redirect(url_for('index'))
    
    employees = Employee.query.all()
    return render_template('admin_dashboard.html', employees=employees)

# Create tables command
@app.cli.command("create_tables")
def create_tables():
    db.create_all()
    print("Tables created!")

# Create admin user command
@app.cli.command("create_admin")
def create_admin():
    admin = Employee.query.filter_by(emp_id='admin').first()
    if not admin:
        temp_password = generate_random_password()
        admin = Employee(emp_id='admin', role='admin')
        admin.set_password(temp_password)
        db.session.add(admin)
        db.session.commit()
        print(f"Admin created with temporary password: {temp_password}")
    else:
        print("Admin already exists")

if __name__ == '__main__':
    app.run(debug=True)