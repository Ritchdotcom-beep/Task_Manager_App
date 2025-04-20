# main_app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

# Employee service configuration
EMPLOYEE_SERVICE_URL = os.environ.get('EMPLOYEE_SERVICE_URL', 'http://localhost:5001/api')
API_KEY = os.environ.get('API_KEY', 'dev_api_key')

# Helper functions for API calls
def api_headers():
    key = os.environ.get('API_KEY', 'dev_api_key')
    print(f"Using API key: {key}")  # Debug line
    return {'X-API-KEY': key, 'Content-Type': 'application/json'}

def get_employee(emp_id):
    response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', headers=api_headers())
    if response.status_code == 200:
        return response.json()
    return None

def authenticate_employee(emp_id, password):
    data = {'emp_id': emp_id, 'password': password}
    headers = api_headers()
    print(f"Sending auth request to {EMPLOYEE_SERVICE_URL}/authenticate")
    print(f"Auth request data: {data}")
    
    try:
        response = requests.post(f'{EMPLOYEE_SERVICE_URL}/authenticate', json=data, headers=headers)
        print(f"Auth response status: {response.status_code}")
        print(f"Auth response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Parsed response: {result}")
            return result
        else:
            print(f"Error response headers: {response.headers}")
            print(f"Error response body: {response.text}")
            return {'authenticated': False, 'reason': f'API error: {response.status_code}'}
    except Exception as e:
        print(f"Exception during authentication: {str(e)}")
        return {'authenticated': False, 'reason': f'Exception: {str(e)}'}

def change_employee_password(emp_id, current_password, new_password):
    data = {
        'emp_id': emp_id,
        'current_password': current_password,
        'new_password': new_password
    }
    response = requests.post(f'{EMPLOYEE_SERVICE_URL}/change_password', json=data, headers=api_headers())
    return response.json() if response.status_code == 200 else None

def get_all_employees():
    response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees', headers=api_headers())
    if response.status_code == 200:
        return response.json()
    return []

def create_new_employee(emp_id, name, email, role):
    data = {
        'emp_id': emp_id,
        'name': name,
        'email': email,
        'role': role
    }
    response = requests.post(f'{EMPLOYEE_SERVICE_URL}/employees', json=data, headers=api_headers())
    if response.status_code == 201:
        return response.json()
    return None

def update_employee(emp_id, data):
    response = requests.put(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', json=data, headers=api_headers())
    if response.status_code == 200:
        return response.json()
    return None

def delete_employee(emp_id):
    response = requests.delete(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', headers=api_headers())
    return response.status_code == 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin_login_page():
    """Special route for admin login that bypasses role selection"""
    return render_template("admin_login.html")

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        password = request.form.get('password')
        
        # Attempt authentication
        result = authenticate_employee(emp_id, password)
        
        if not result['authenticated']:
            flash('Invalid employee ID or password')
            return render_template('admin_login.html')
        
        employee = result['employee']
        
        if employee['role'] != 'admin':
            flash('This employee ID is not registered as an admin')
            return render_template('admin_login.html')
        
        # Store employee info in session
        session['emp_id'] = emp_id
        session['role'] = employee['role']
        session['name'] = employee['name']
        session['email'] = employee['email']
        
        # If first login, redirect to password change page
        if employee['is_first_login']:
            return redirect(url_for('change_password'))
        
        # Direct to admin dashboard
        return redirect(url_for('admin_dashboard'))
    
    return render_template('admin_login.html')

@app.route('/select_role/<role>')
def select_role(role):
    if role not in ['developer', 'project_manager', 'human_resource']:
        flash('Invalid role selected')
        return redirect(url_for('index'))
    
    session['selected_role'] = role
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'selected_role' not in session:
        flash('Please select a role first')
        return redirect(url_for('index'))
    
    role = session['selected_role']
    print(f"\n--- LOGIN ATTEMPT ---\nRole selected: {role}")

    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        password = request.form.get('password')
        print(f"Attempting login for: {emp_id}")

        result = authenticate_employee(emp_id, password)
        print(f"Auth result: {result}")

        if not result or not result.get('authenticated'):
            flash('Invalid employee ID or password')
            return render_template('login.html', role=role)

        employee = result['employee']
        print(f"Employee data received: {employee}")

        # Case-insensitive role comparison
        if employee['role'].lower() != role.lower():
            flash(f'This employee ID is not registered as a {role.replace("_", " ")}')
            return render_template('login.html', role=role)

        # Store session data
        session.update({
            'emp_id': emp_id,
            'role': employee['role'],
            'name': employee['name'],
            'email': employee['email']
        })

        # Debug print session data
        print(f"\n--- SESSION DATA ---\n{session}\n")

        # Handle first login (using get() with True default)
        first_login = employee.get('is_first_login', True)
        print(f"First login status: {first_login} (Type: {type(first_login)})")

        if first_login:
            print("Redirecting to change_password")
            return redirect(url_for('change_password'))

        # Regular redirect based on role
        print(f"Redirecting to {role}_dashboard")
        return redirect(url_for(f'{role}_dashboard'))

    return render_template('login.html', role=role)

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'emp_id' not in session:
        return redirect(url_for('index'))
    
    # Add debug logging
    print(f"Reached change_password route for employee {session['emp_id']}")
    
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('New passwords do not match')
            return render_template('change_password.html')
        
        # Call API to change password
        result = change_employee_password(session['emp_id'], current_password, new_password)
        
        if not result or not result.get('success'):
            flash('Current password is incorrect')
            return render_template('change_password.html')
        
        flash('Password changed successfully')
        
        # Redirect based on role
        role = session['role'].lower().replace(' ', '_')
        if role == 'developer':
            return redirect(url_for('developer_dashboard'))
        elif role == 'project_manager':
            return redirect(url_for('manager_dashboard'))
        elif role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('hr_dashboard'))
    
    # Get employee to check first_login status
    employee = get_employee(session['emp_id'])
    first_login = employee['is_first_login'] if employee else False
    
    print(f"Employee first_login status: {first_login}")
    
    return render_template('change_password.html', first_login=first_login)

@app.route('/developer_dashboard')
def developer_dashboard():
    if 'emp_id' not in session:
        return redirect(url_for('index'))
    normalized_role = session['role'].lower().replace(' ', '_')
    if normalized_role != 'developer':
        return redirect(url_for('index'))
    return render_template('developer_dashboard.html', employee=session)

@app.route('/manager_dashboard')
def manager_dashboard():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return redirect(url_for('index'))
    return render_template('manager_dashboard.html', employee=session)

@app.route('/hr_dashboard')
def hr_dashboard():
    if 'emp_id' not in session or session['role'] != 'human resource':
        return redirect(url_for('index'))
    return render_template('hr_dashboard.html', employee=session)

@app.route('/admin/create_employee', methods=['GET', 'POST'])
def create_employee():
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        name = request.form.get('name')
        email = request.form.get('email')
        role = request.form.get('role')
        
        # Call API to create employee
        result = create_new_employee(emp_id, name, email, role)
        
        if not result:
            flash('Failed to create employee')
            return render_template('create_employee.html')
        
        flash(f'Employee created with ID: {emp_id} and temporary password: {result["temp_password"]}')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('create_employee.html')

@app.route('/admin/edit_employee/<emp_id>', methods=['GET', 'POST'])
def edit_employee(emp_id):
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    employee = get_employee(emp_id)
    if not employee:
        flash('Employee not found')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        data = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'role': request.form.get('role')
        }
        
        # Update employee via API
        result = update_employee(emp_id, data)
        
        if not result:
            flash('Failed to update employee')
            return render_template('edit_employee.html', employee=employee)
        
        flash('Employee updated successfully')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('edit_employee.html', employee=employee)

@app.route('/admin/delete_employee/<emp_id>', methods=['POST'])
def admin_delete_employee(emp_id):
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    if emp_id == session['emp_id']:
        flash('Cannot delete your own account')
        return redirect(url_for('admin_dashboard'))
    
    # Delete employee via API
    if delete_employee(emp_id):
        flash('Employee deleted successfully')
    else:
        flash('Failed to delete employee')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'emp_id' not in session or session['role'] != 'admin':
        return redirect(url_for('index'))
    
    # Get all employees from API
    employees = get_all_employees()
    
    return render_template('admin_dashboard.html', employees=employees, current_user=session)

@app.route('/debug_employee/<emp_id>')
def debug_employee(emp_id):
    employee = get_employee(emp_id)
    if not employee:
        return "Employee not found", 404
    return jsonify({
        'data': employee,
        'is_first_login_type': str(type(employee.get('is_first_login'))),
        'role_type': str(type(employee.get('role')))
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)