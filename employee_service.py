# employee_service.py

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import string
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/employee_db')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_for_api')

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Updated Employee Model with name and email
class Employee(db.Model):
    __tablename__ = 'employees'
    
    emp_id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # Added name field
    email = db.Column(db.String(100), nullable=False)  # Added email field
    password_hash = db.Column(db.String(255), nullable=False)
    is_first_login = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Employee {self.emp_id} - {self.name}>'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
        'emp_id': self.emp_id,
        'name': self.name,
        'email': self.email,
        'role': self.role.strip().lower(),  # Simplified role format
        'is_first_login': bool(self.is_first_login),  # Ensure boolean
        'created_at': self.created_at.isoformat() if self.created_at else None,
        'last_login': self.last_login.isoformat() if self.last_login else None
    }
# Function to generate random password
def generate_random_password(length=4):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

# API Authentication - Simple API key check
def authenticate_request():
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.environ.get('API_KEY', 'dev_api_key')
    
    print(f"Received headers: {dict(request.headers)}")
    print(f"Received API key: '{api_key}'")
    print(f"Expected API key: '{expected_key}'")
    print(f"Keys equal? {api_key == expected_key}")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")
    
    if not api_key or api_key != expected_key:
        return False
    return True

# API Routes
@app.route('/api/employees', methods=['GET'])
def get_all_employees():
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employees = Employee.query.all()
    return jsonify([employee.to_dict() for employee in employees])

@app.route('/api/employees/<emp_id>', methods=['GET'])
def get_employee(emp_id):
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    return jsonify(employee.to_dict())

@app.route('/api/employees', methods=['POST'])
def create_employee():
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    
    # Validate required fields
    required_fields = ['emp_id', 'name', 'email', 'role']
    if not all(k in data for k in required_fields):
        return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400
    
    # Check if employee already exists
    existing_employee = Employee.query.get(data['emp_id'])
    if existing_employee:
        return jsonify({'error': 'Employee ID already exists'}), 409
    
    # Check if email already exists
    existing_email = Employee.query.filter_by(email=data['email']).first()
    if existing_email:
        return jsonify({'error': 'Email already exists'}), 409
    
    # Generate temporary password
    temp_password = data.get('password', generate_random_password())
    
    # Create new employee
    new_employee = Employee(
        emp_id=data['emp_id'],
        name=data['name'],
        email=data['email'],
        role=data['role'],
        is_first_login=data.get('is_first_login', True)
    )
    new_employee.set_password(temp_password)
    
    db.session.add(new_employee)
    try:
        db.session.commit()
        response = new_employee.to_dict()
        # Only return the password in the response, don't store it
        response['temp_password'] = temp_password
        return jsonify(response), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees/<emp_id>', methods=['PUT'])
def update_employee(emp_id):
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    data = request.json
    
    # Update fields
    if 'name' in data:
        employee.name = data['name']
    if 'email' in data:
        # Check if email already exists on another account
        existing_email = Employee.query.filter_by(email=data['email']).first()
        if existing_email and existing_email.emp_id != emp_id:
            return jsonify({'error': 'Email already exists'}), 409
        employee.email = data['email']
    if 'role' in data:
        employee.role = data['role']
    if 'is_first_login' in data:
        employee.is_first_login = data['is_first_login']
    if 'password' in data:
        employee.set_password(data['password'])
    
    try:
        db.session.commit()
        return jsonify(employee.to_dict()), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees/<emp_id>', methods=['DELETE'])
def delete_employee(emp_id):
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    try:
        db.session.delete(employee)
        db.session.commit()
        return jsonify({'message': f'Employee {emp_id} deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_employee():
    print(f"Request received at /api/authenticate")
    
    if not authenticate_request():
        print("API key authentication failed")
        return jsonify({'error': 'Unauthorized', 'message': 'API key authentication failed'}), 401
    
    print("API key authentication successful")
    
    try:
        data = request.json
        print(f"Received authentication data: {data}")
        
        # Validate required fields
        if not all(k in data for k in ['emp_id', 'password']):
            print("Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400
        
        employee = Employee.query.get(data['emp_id'])
        print(f"Found employee: {employee}")
        
        if not employee:
            print("Employee not found")
            return jsonify({'authenticated': False, 'reason': 'Employee not found'}), 401
        
        password_check = employee.check_password(data['password'])
        print(f"Password check result: {password_check}")
        
        if not password_check:
            print("Invalid password")
            return jsonify({'authenticated': False, 'reason': 'Invalid password'}), 401
        
        # Update last login time
        employee.last_login = datetime.utcnow()
        db.session.commit()
        
        result = {
            'authenticated': True,
            'employee': employee.to_dict()
        }
        print(f"Authentication successful: {result}")
        
        return jsonify(result)
    except Exception as e:
        print(f"Exception during authentication: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500
    
@app.route('/api/change_password', methods=['POST'])
def change_password():
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    
    # Validate required fields
    if not all(k in data for k in ['emp_id', 'current_password', 'new_password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    employee = Employee.query.get(data['emp_id'])
    
    if not employee or not employee.check_password(data['current_password']):
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    employee.set_password(data['new_password'])
    employee.is_first_login = False
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'employee': employee.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# Command to create tables
@app.cli.command("create_tables")
def create_tables():
    db.create_all()
    print("Tables created!")

# Command to create admin user
@app.cli.command("create_admin")
def create_admin():
    admin = Employee.query.filter_by(emp_id='admin').first()
    if not admin:
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
    else:
        print("Admin already exists")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)