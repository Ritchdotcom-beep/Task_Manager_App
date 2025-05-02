# main_app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import requests
import json
from dotenv import load_dotenv
from email_services import send_credentials_email
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

# Employee service configuration
EMPLOYEE_SERVICE_URL = os.environ.get('EMPLOYEE_SERVICE_URL', 'http://localhost:5001/api')
API_KEY = os.environ.get('API_KEY', 'dev_api_key')
PROJECT_TYPES = {
    "website_development": ["HTML", "CSS", "JavaScript", "React", "Vue", "Angular"],
    "mobile_app_development": ["Swift", "Kotlin", "React Native", "Flutter"],
    "machine_learning": ["Python", "TensorFlow", "PyTorch", "Scikit-learn"],
    # Add other project types as needed
}
# Add to main_app.py
TASK_SERVICE_URL = os.environ.get('TASK_SERVICE_URL', 'http://localhost:5002/api')

def assign_tasks(tasks):
    response = requests.post(
        f'{TASK_SERVICE_URL}/task-service/assign-tasks',
        json={'tasks': tasks},
        headers=api_headers()
    )
    if response.status_code == 200:
        return response.json()
    return None

def get_project_types():
    """Get all project types from task service with fallback to local definition"""
    try:
        response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/project-types',
            headers=api_headers()
        )
        if response.status_code == 200:
            return response.json().get('project_types', []), response.json().get('project_type_details', {})
        return list(PROJECT_TYPES.keys()), PROJECT_TYPES
    except Exception as e:
        print(f"Error fetching project types: {str(e)}")
        return list(PROJECT_TYPES.keys()), PROJECT_TYPES

def get_skills_for_project_type(project_type):
    """Get required skills for a specific project type"""
    try:
        response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/skills-for-project',
            params={'project_type': project_type},
            headers=api_headers()
        )
        if response.status_code == 200:
            return response.json().get('skills', [])
        return PROJECT_TYPES.get(project_type, [])
    except Exception as e:
        print(f"Error fetching skills for project type: {str(e)}")
        return PROJECT_TYPES.get(project_type, [])

def format_date(iso_date):
    """Format ISO date string to human-readable format"""
    if not iso_date:
        return "Not set"
    try:
        date_obj = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
        return date_obj.strftime("%b %d, %Y %H:%M")
    except:
        return iso_date

def get_developer_tasks(emp_id):
    """Get all tasks assigned to a specific developer"""
    try:
        response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks',
            params={'emp_id': emp_id},
            headers=api_headers()
        )
        if response.status_code == 200:
            return response.json().get('tasks', [])
        return []
    except Exception as e:
        print(f"Error fetching developer tasks: {str(e)}")
        return []

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

def create_new_employee(name, email, role, skills=None, experience=None, emp_id=None):
    data = {
        'name': name,
        'email': email,
        'role': role
    }
    
    # Only include emp_id if it's provided and not empty
    if emp_id:
        data['emp_id'] = emp_id
        
    # Include skills if provided
    if skills:
        data['skills'] = skills
        
    # Include experience if provided
    if experience is not None:
        data['experience'] = experience
    
    response = requests.post(
        f'{EMPLOYEE_SERVICE_URL}/employees',
        json=data,
        headers=api_headers()
    )
    
    if response.status_code == 201:
        result = response.json()
        return True, result.get('employee', {}).get('emp_id', '')
    else:
        error_message = response.json().get('error', 'Failed to create employee')
        return False, error_message

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
        if employee['role'].lower().replace(' ', '_') != role.lower():
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
            return redirect(url_for('project_manager_dashboard'))
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
        flash('Please log in to access the developer dashboard', 'warning')
        return redirect(url_for('login'))
    
    normalized_role = session.get('role', '').lower().replace(' ', '_')
    if normalized_role != 'developer':
        flash('Access restricted to developers only', 'danger')
        return redirect(url_for('index'))
    
    # Fetch complete employee data
    employee_data = get_employee(session['emp_id'])
    
    # Fallback with defaults if needed
    if not employee_data:
        employee_data = dict(session)
        employee_data['success_rate'] = 0.0
        employee_data['tasks_completed'] = 0
        employee_data['experience'] = 0
        employee_data['skills'] = []
    
    # Get assigned tasks for this developer
    assigned_tasks = get_developer_tasks(session['emp_id'])
    
    # Get dashboard statistics for the developer
    try:
        dashboard_response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/dashboard?emp_id={session["emp_id"]}',
            headers=api_headers()
        )
        dashboard_data = dashboard_response.json() if dashboard_response.status_code == 200 else {}
    except Exception as e:
        print(f"Error getting dashboard data: {str(e)}")
        dashboard_data = {}
    
    # Categorize tasks by status
    in_progress_tasks = [task for task in assigned_tasks if task.get('status') == 'in_progress']
    pending_tasks = [task for task in assigned_tasks if task.get('status') == 'assigned']
    completed_tasks = [task for task in assigned_tasks if task.get('status') == 'completed']
    
    # Get project types for filtering
    try:
        project_types_response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/project-types',
            headers=api_headers()
        )
        project_types = project_types_response.json().get('project_types', []) if project_types_response.status_code == 200 else []
    except Exception as e:
        print(f"Error getting project types: {str(e)}")
        project_types = []
    
    return render_template(
        'developer_dashboard.html', 
        employee=employee_data,
        tasks=assigned_tasks,
        in_progress_tasks=in_progress_tasks,
        pending_tasks=pending_tasks,
        completed_tasks=completed_tasks,
        dashboard=dashboard_data,
        project_types=project_types,
        format_date=format_date
    )


@app.route('/project_manager_dashboard')
def project_manager_dashboard():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return redirect(url_for('index'))
    
    # Fetch complete employee data from the employee service
    employee_data = get_employee(session['emp_id'])
    
    # If employee data couldn't be fetched, use session data as fallback with defaults
    if not employee_data:
        employee_data = dict(session)
        employee_data['success_rate'] = 0.0
        employee_data['tasks_completed'] = 0
        employee_data['experience'] = 0
    
    # Get all employees for team stats
    all_employees = get_all_employees()
    
    # Calculate team statistics
    team_stats = {
        'total_members': len(all_employees),
        'active_tasks': 0,  # You would get this from your task service
        'avg_success': sum(emp.get('success_rate', 0) for emp in all_employees) / len(all_employees) if all_employees else 0,
        'top_skills': {},
        'top_performers': sorted(
            [emp for emp in all_employees if emp.get('tasks_completed', 0) > 0],
            key=lambda x: x.get('success_rate', 0),
            reverse=True
        )[:3]  # Top 3 performers
    }
    
    # Calculate skill distribution
    skill_counts = {}
    for emp in all_employees:
        for skill in emp.get('skills', []):
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    team_stats['top_skills'] = dict(sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)[:5])  # Top 5 skills
    
    # Get project types from task service
    project_types, project_type_details = get_project_types()
    
    return render_template(
        'project_manager_dashboard.html',
        employee=employee_data,
        team_stats=team_stats,
        project_types=project_types,
        project_type_details=project_type_details
    )

@app.route('/task_management')
def task_management():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return redirect(url_for('index'))
    
    # Get project types and their details
    project_types, project_type_details = get_project_types()
    
    # Get all employees for assignment dropdown
    employees = get_all_employees()
    
    return render_template(
        'task_management.html',
        project_types=project_types,
        project_type_details=project_type_details,
        employees=employees
    )

    
    if not result or 'error' in result:
        return jsonify({'success': False, 'error': 'Failed to assign task'}), 500
        
    return jsonify({
        'success': True,
        'task': task_data,
        'assignment': result.get('assignments', {}).get(task_data['task_id'])
    })


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
        # Employee ID is now optional
        emp_id = request.form.get('emp_id', '').strip()
        name = request.form.get('name')
        email = request.form.get('email')
        role = request.form.get('role')
        
        # Get skills as a list from the comma-separated input
        skills_input = request.form.get('skills', '').strip()
        skills = [skill.strip() for skill in skills_input.split(',')] if skills_input else None
        
        # Get experience as an integer
        experience_input = request.form.get('experience', '').strip()
        experience = int(experience_input) if experience_input and experience_input.isdigit() else None
        
        # Call API to create employee with the new parameters
        success, result = create_new_employee(
            name=name,
            email=email,
            role=role,
            skills=skills,
            experience=experience,
            emp_id=emp_id if emp_id else None
        )
        
        if not success:
            flash(f'Failed to create employee: {result}')
        else:
            # Get the assigned employee ID for the success message
            assigned_emp_id = result if isinstance(result, str) else emp_id
            flash(f'Employee created with ID: {assigned_emp_id}. Credentials sent via email.')
        
        return redirect(url_for('admin_dashboard'))
    
    # Get project types for skill suggestions
    project_types, project_type_details = get_project_types()
    all_skills = []
    for skills_list in project_type_details.values():
        all_skills.extend(skills_list)
    # Remove duplicates while preserving order
    unique_skills = list(dict.fromkeys(all_skills))
    
    return render_template('create_employee.html', skills=unique_skills)

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
@app.route('/admin/update_metrics/<emp_id>', methods=['POST'])
def update_metrics(emp_id):
    if 'emp_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized access'}), 403
    
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Prepare update data
    update_data = {}
    if 'tasks_completed' in data:
        update_data['tasks_completed'] = data['tasks_completed']
    if 'success_rate' in data:
        update_data['success_rate'] = data['success_rate']
    
    # Update employee via API
    result = update_employee(emp_id, update_data)
    
    if not result:
        return jsonify({'success': False, 'error': 'Failed to update employee metrics'}), 500
    
    return jsonify({'success': True})

# Add this endpoint to main_app.py - ensure it is placed before the 'if __name__ == '__main__':' line

@app.route('/api/create_task', methods=['POST'])
def create_task():
    """Create a new task and get assignment recommendations"""
    if 'emp_id' not in session or session.get('role') != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    task_data = request.json
    # Validate task data
    required_fields = ['task_id', 'project_type', 'complexity', 'priority']
    for field in required_fields:
        if field not in task_data:
            return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
    
    # Get skills for project type if not provided
    if 'skills' not in task_data or not task_data['skills']:
        task_data['skills'] = get_skills_for_project_type(task_data['project_type'])
    
    # Create list of tasks (API expects an array)
    tasks = [task_data]
    
    # Assign task using the task service
    result = assign_tasks(tasks)
    
    if not result or 'success' not in result or not result['success']:
        return jsonify({
            'success': False, 
            'error': result.get('error', 'Failed to assign task')
        }), 500
        
    return jsonify({
        'success': True,
        'task': task_data,
        'assignment': result.get('assignments', {}).get(task_data['task_id'])
    })

# Add this JavaScript fetch endpoint for the task management page
# Fix for the get_assignment_recommendation function in main_app.py

@app.route('/api/get_assignment_recommendation', methods=['POST'])
def get_assignment_recommendation():
    """Get assignment recommendation for a task without saving it"""
    if 'emp_id' not in session or session.get('role') != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    task_data = request.json
    tasks = [task_data]
    
    # Get skills for project type if not provided
    if 'skills' not in task_data or not task_data['skills']:
        task_data['skills'] = get_skills_for_project_type(task_data['project_type'])
    
    # Get recommendation without persistence
    try:
        result = assign_tasks(tasks)
        
        if not result:
            return jsonify({
                'success': False, 
                'error': 'Failed to get recommendation - result is None'
            }), 500
        
        if 'success' not in result or not result['success']:
            return jsonify({
                'success': False, 
                'error': result.get('error', 'Failed to get recommendation')
            }), 500
            
        return jsonify({
            'success': True,
            'recommendations': result.get('assignments', {}).get(task_data['task_id'])
        })
    except Exception as e:
        import traceback
        print(f"Error in get_assignment_recommendation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Exception: {str(e)}'
        }), 500

@app.route('/update_task_status', methods=['POST'])
def update_task_status():
    """Update a task's status"""
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
    data = request.json
    if not data or 'task_id' not in data or 'status' not in data:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
    task_id = data['task_id']
    new_status = data['status']
    rating = data.get('rating')
    emp_id = session['emp_id']
    
    # Print URL for debugging
    task_url = f'{TASK_SERVICE_URL}/task-service/task/{task_id}/status'
    print(f"Making request to: {task_url}")
        
    # Call the task service API to update the task
    try:
        # Simpler URL without '/api/' prefix to avoid duplication
        response = requests.put(
            task_url,
            json={
                'emp_id': emp_id,
                'status': new_status,
                'rating': rating
            },
            headers=api_headers()
        )
            
        # Print response for debugging
        print(f"Update task response: {response.status_code} - {response.text}")
            
        if response.status_code != 200:
            return jsonify({
                'success': False, 
                'error': f'Task service error: {response.status_code} - {response.text}'
            }), 500
                
        result = response.json()
        return jsonify({
            'success': True,
            'task': result.get('task', {})
        })
        
    except Exception as e:
        import traceback
        print(f"Error updating task status: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Exception: {str(e)}'
        }), 500

# Add a jinja template filter for formatting dates
@app.template_filter('datetime')
def format_datetime(value, format='%B %d, %Y %I:%M %p'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except:
            return value
    return value.strftime(format)

@app.route('/test_email')
def test_email():
    test_email = "lolomoakamela@gmail.com"  # Use a real email you can check
    result = send_credentials_email(
        email=test_email,
        emp_id="TEST123",
        temp_password="testpass123"
    )
    return jsonify({"success": result})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)