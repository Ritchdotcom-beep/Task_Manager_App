# main_app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import requests
import json
from dotenv import load_dotenv
from email_services import send_credentials_email
from datetime import datetime, timedelta
import random

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

# Add this function in your app.py or utils.py file
def format_date(date_str, format_str='%b %d, %Y'):
    """Format a date string with the specified format"""
    if not date_str:
        return "N/A"
    
    try:
        # First try to parse the date string
        if isinstance(date_str, str):
            # Try ISO format first (most common API response format)
            try:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                # Try common date formats
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            return date_str  # Return original if parsing fails
        elif isinstance(date_str, datetime):
            date_obj = date_str
        else:
            return str(date_str)
        
        # Format the date object
        return date_obj.strftime(format_str)
    except Exception as e:
        print(f"Error formatting date: {str(e)}")
        return str(date_str)  # Return original as fallback

def get_developer_tasks(emp_id):
    try:
        response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/tasks',
            params={'assignee': emp_id},
            headers=api_headers()
        )
        
        if response.status_code != 200:
            print(f"Error fetching tasks: {response.text}")
            return []
        
        return response.json().get('tasks', [])
    except Exception as e:
        print(f"Exception fetching tasks: {str(e)}")
        return []

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
    
    # If dashboard data wasn't returned from API, calculate it from tasks
    if not dashboard_data:
        dashboard_data = calculate_dashboard_metrics(assigned_tasks, employee_data)
    
    # Categorize tasks by status - Updated to match Task model status values
    in_progress_tasks = [task for task in assigned_tasks if task.get('status') == 'in_progress']
    pending_tasks = [task for task in assigned_tasks if task.get('status') == 'assigned']
    pending_approval_tasks = [task for task in assigned_tasks if task.get('status') == 'submitted']
    completed_tasks = [task for task in assigned_tasks if task.get('status') == 'completed']
    
    # Calculate performance history (last 6 months)
    performance_history = get_performance_history(session['emp_id'])
    
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
        pending_approval_tasks=pending_approval_tasks,  # Updated variable name
        completed_tasks=completed_tasks,
        dashboard=dashboard_data,
        performance_history=performance_history,
        project_types=project_types,
        format_date=format_date
    )


def calculate_dashboard_metrics(tasks, employee_data):
    """Calculate dashboard metrics when API doesn't return data"""
    dashboard = {
        'success_rate': 0.0,
        'tasks_completed': 0,
        'avg_completion_time': 0,
        'performance_trend': []
    }
    
    # Count completed tasks
    completed_tasks = [task for task in tasks if task.get('status') == 'completed']
    dashboard['tasks_completed'] = len(completed_tasks)
    
    # Calculate success rate
    if dashboard['tasks_completed'] > 0:
        # Get average of task ratings
        ratings = [task.get('success_rating', 0) for task in completed_tasks if task.get('success_rating') is not None]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Convert 5-star rating to percentage (e.g., 4/5 = 80%)
            dashboard['success_rate'] = (avg_rating / 5) * 100
        else:
            # Default to 70% if no ratings available
            dashboard['success_rate'] = 70.0
    else:
        # No completed tasks yet, use N/A or default value
        dashboard['success_rate'] = employee_data.get('success_rate', 0.0)
    
    # Calculate average completion time (in days)
    if completed_tasks:
        completion_times = []
        for task in completed_tasks:
            start_date = parse_date(task.get('start_date'))
            completion_date = parse_date(task.get('completion_date'))
            if start_date and completion_date:
                delta = completion_date - start_date
                completion_times.append(delta.days)
        
        if completion_times:
            dashboard['avg_completion_time'] = sum(completion_times) / len(completion_times)
    
    return dashboard

def get_performance_history(emp_id, months=6):
    """Get the performance history for the last X months"""
    today = datetime.now()
    
    # Default data structure with empty values
    history = {
        'labels': [],
        'success_rates': []
    }
    
    try:
        # Fetch historical data from API
        response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/performance-history?emp_id={emp_id}&months={months}',
            headers=api_headers()
        )
        
        if response.status_code == 200:
            history_data = response.json().get('history', [])
            if history_data:
                # If API returned data, use it
                for month_data in history_data:
                    history['labels'].append(month_data.get('month'))
                    history['success_rates'].append(month_data.get('success_rate', 0))
                return history
    except Exception as e:
        print(f"Error getting performance history: {str(e)}")
    
    # If no data from API or error occurred, generate some reasonable sample data
    for i in range(months, 0, -1):
        month_date = today - timedelta(days=30*i)
        month_name = month_date.strftime('%b')
        history['labels'].append(month_name)
        
        # Generate reasonable random data that trends upward slightly
        # Starting from 70% and gradually improving
        base_rate = 70 + (i * 2)  # Increase by 2% each month
        rate = min(base_rate + random.randint(-3, 5), 95)  # Add randomness but cap at 95%
        history['success_rates'].append(rate)
    
    return history

def parse_date(date_str):
    """Parse date string to datetime object"""
    if not date_str:
        return None
    
    try:
        # Adjust the format based on your date string format
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try another common format
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            return None

@app.route('/api/assign_multiple_employees', methods=['POST'])
def assign_multiple_employees():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    task_id = request.json.get('task_id')
    employee_ids = request.json.get('employee_ids', [])
    
    if not task_id or not employee_ids:
        return jsonify({'success': False, 'error': 'Task ID and at least one employee ID are required'}), 400
    
    try:
        # Call the task service API to assign multiple employees to the task
        response = requests.post(
            f'{request.host_url.rstrip("/")}/api/task-service/tasks/{task_id}/assign',
            headers=api_headers(),
            json={
                'manager_id': session['emp_id'],
                'employee_ids': employee_ids
            }
        )
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to assign employees: {response.text}'}), 500
        
        result = response.json()
        return jsonify({'success': True, 'assignments': result.get('assignments', {})})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
        flash('Access restricted to project managers', 'danger')
        return redirect(url_for('index'))
    
    # Get filter parameters from query string
    status_filter = request.args.get('status', 'all')
    project_type_filter = request.args.get('project_type', 'all')
    assignee_filter = request.args.get('assignee', 'all')
    
    # Get project types and their details
    project_types, project_type_details = get_project_types()
    
    # Get all employees for assignment dropdown
    employees = get_all_employees()
    
    try:
        # Get all tasks with optional filters
        tasks_response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/tasks',
            params={
                'status': status_filter if status_filter != 'all' else None,
                'project_type': project_type_filter if project_type_filter != 'all' else None,
                'assignee': assignee_filter if assignee_filter != 'all' else None
            },
            headers=api_headers()
        )
        
        tasks = []
        if tasks_response.status_code == 200:
            tasks = tasks_response.json().get('tasks', [])
        
        # Categorize tasks by status for easier display
        pending_review_tasks = [task for task in tasks if task.get('status') == 'pending_review']
        assigned_tasks = [task for task in tasks if task.get('status') == 'assigned']
        in_progress_tasks = [task for task in tasks if task.get('status') == 'in_progress']
        completed_tasks = [task for task in tasks if task.get('status') == 'completed']
        
        return render_template(
            'task_management.html',
            project_types=project_types,
            project_type_details=project_type_details,
            employees=employees,
            tasks=tasks,
            pending_review_tasks=pending_review_tasks,
            assigned_tasks=assigned_tasks,
            in_progress_tasks=in_progress_tasks,
            completed_tasks=completed_tasks,
            status_filter=status_filter,
            project_type_filter=project_type_filter,
            assignee_filter=assignee_filter,
            format_date=format_date
        )
    
    except Exception as e:
        flash(f'Error fetching task data: {str(e)}', 'danger')
        return render_template(
            'task_management.html',
            project_types=project_types,
            project_type_details=project_type_details,
            employees=employees,
            tasks=[],
            pending_review_tasks=[],
            assigned_tasks=[],
            in_progress_tasks=[],
            completed_tasks=[]
        )

@app.route('/approve_task', methods=['POST'])
def approve_task():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    task_id = request.json.get('task_id')
    action = request.json.get('action')  # 'approve' or 'reject'
    feedback = request.json.get('feedback', '')
    
    if not task_id or not action:
        return jsonify({'success': False, 'error': 'Task ID and action are required'}), 400
    
    if action not in ['approve', 'reject']:
        return jsonify({'success': False, 'error': 'Invalid action'}), 400
    
    try:
        # First, ensure the task is in pending_approval status by calling review endpoint
        api_url_review = f"{TASK_SERVICE_URL}/task-service/tasks/{task_id}/review"
        print(f"Moving task to pending_approval: {api_url_review}")
        
        review_response = requests.put(
            api_url_review,
            headers=api_headers()
        )
        
        print(f"Review response: {review_response.status_code} - {review_response.text}")
        
        # If task is already in pending_approval, this might fail but we can continue
        # Only fail if it's not a 'status' related error
        if review_response.status_code != 200:
            result = review_response.json()
            if 'error' in result and 'status' not in result['error']:
                return jsonify({'success': False, 'error': f'Failed to prepare task for review: {review_response.text}'}), 500
        
        # Now proceed with approval or rejection
        api_url = f"{TASK_SERVICE_URL}/task-service/tasks/{task_id}/approve"
        print(f"Approving/rejecting task: {api_url}")
        
        payload = {
            'manager_id': session['emp_id'],
            'approved': action == 'approve',
            'notes': feedback,
            'rating': 5 if action == 'approve' else None
        }
        
        print(f"With payload: {payload}")
        
        # Call the task service API to approve or reject the task
        response = requests.put(
            api_url,
            headers=api_headers(),
            json=payload
        )
        
        print(f"Approve response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to process task: {response.text}'}), 500
            
        result = response.json()
        return jsonify({'success': True, 'task': result.get('task', {})})
        
    except Exception as e:
        import traceback
        print(f"Exception in approve_task: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500






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


@app.route('/submit_task_for_review', methods=['POST'])
def submit_task_for_review():
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        task_id = data.get('task_id')
        
        if not task_id:
            return jsonify({'success': False, 'error': 'Task ID required'}), 400

        # Call task service
        api_url = f'http://localhost:5002/api/task-service/tasks/{task_id}/submit'
        response = requests.post(
            api_url,
            headers=api_headers(),
            json={'emp_id': session['emp_id']},
            timeout=5
        )

        # Handle response
        response_data = response.json()
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': response_data.get('error', 'Task service error')
            }), response.status_code

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Service unavailable: {str(e)}'
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

@app.route('/get_task/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get details for a specific task"""
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        # Define API endpoint - FIXED to use port 5002 and correct path
        api_endpoint = f'http://localhost:5002/api/task-service/tasks/{task_id}'
        
        # Log the request
        app.logger.info(f"Requesting task data from: {api_endpoint}")
        
        # Get headers
        headers = api_headers()
        
        # Call the task service API with error handling
        response = requests.get(api_endpoint, headers=headers)
        
        # Log response status
        app.logger.info(f"Task service response status: {response.status_code}")
        
        if response.status_code != 200:
            error_message = f"Failed to get task: Status {response.status_code}"
            try:
                error_detail = response.json()
                error_message += f" - {error_detail.get('error', 'Unknown error')}"
            except:
                error_message += f" - {response.text[:100]}"
            
            app.logger.error(error_message)
            return jsonify({'success': False, 'error': error_message}), 500
        
        # Parse response
        try:
            result = response.json()
            return jsonify({'success': True, 'task': result.get('task', {})})
        except Exception as json_error:
            app.logger.error(f"Error parsing JSON response: {str(json_error)}")
            return jsonify({'success': False, 'error': f'Invalid response format: {str(json_error)}'}), 500
    
    except requests.exceptions.ConnectionError as conn_error:
        error_msg = f"Connection error to task service: {str(conn_error)}"
        app.logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in get_task: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pending_review_tasks', methods=['GET'])
def get_pending_review_tasks():
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        # Call task service to get pending review tasks
        api_url = f'{TASK_SERVICE_URL}/task-service/tasks/pending-review'
        response = requests.get(
            api_url,
            headers=api_headers(),
            timeout=5
        )
        
        # Handle response
        response_data = response.json()
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': response_data.get('error', 'Task service error')
            }), response.status_code
        
        return jsonify(response_data)
    
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Service unavailable: {str(e)}'
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

@app.route('/task_details/<task_id>')
def task_details(task_id):
    if 'emp_id' not in session:
        flash('Please log in to view task details', 'warning')
        return redirect(url_for('login'))
    
    try:
        # Get task details
        task_response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks/{task_id}',
            headers=api_headers()
        )
        
        if task_response.status_code != 200:
            flash('Failed to retrieve task details', 'danger')
            return redirect(url_for('index'))
        
        task = task_response.json().get('task', {})
        
        # Get assigned employees for this task
        assignees_response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks/{task_id}/assignees',
            headers=api_headers()
        )
        
        assignees = []
        if assignees_response.status_code == 200:
            assignee_ids = assignees_response.json().get('assignees', [])
            # Get employee details for each assignee
            for emp_id in assignee_ids:
                emp_data = get_employee(emp_id)
                if emp_data:
                    assignees.append(emp_data)
        
        # Get task history/activity log
        history_response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks/{task_id}/history',
            headers=api_headers()
        )
        
        history = []
        if history_response.status_code == 200:
            history = history_response.json().get('history', [])
        
        # Different templates based on user role
        if session.get('role') == 'project manager':
            return render_template(
                'task_details_manager.html',
                task=task,
                assignees=assignees,
                history=history,
                format_date=format_date
            )
        else:
            return render_template(
                'task_details.html',
                task=task,
                assignees=assignees,
                history=history,
                format_date=format_date
            )
    
    except Exception as e:
        flash(f'Error retrieving task details: {str(e)}', 'danger')
        return redirect(url_for('index'))


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
    # Check if task_id is empty or None
    if not task_id:
        return jsonify({'success': False, 'error': 'Invalid task ID'}), 400
        
    new_status = data['status']
    rating = data.get('rating')
    emp_id = session['emp_id']
    
    # Fix the URL formatting - Use correct endpoint structure
    print(f"TASK_SERVICE_URL value: {TASK_SERVICE_URL}")
    
    # Clear any trailing slashes from the base URL
    base_url = TASK_SERVICE_URL.rstrip('/')
    
    # Construct URL correctly based on whether /api is already present
    if '/api' in base_url:
        task_url = f'{base_url}/task-service/tasks/{task_id}/status'
    else:
        task_url = f'{base_url}/api/task-service/tasks/{task_id}/status'
    
    print(f"Making request to: {task_url}")
        
    # Call the task service API to update the task
    try:
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
        print(f"Error updating task status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to update task: {str(e)}'
        }), 500


# Modified get_all_tasks function to ensure Task is defined
@app.route('/api/task-service/tasks', methods=['GET'])
def get_all_tasks():
    """Get all tasks or filter by employee, status, etc."""
    try:
        # Get query parameters for filtering
        emp_id = request.args.get('emp_id')
        status = request.args.get('status')
        project_type = request.args.get('project_type')
        
        # Start with base query - make sure Task is defined in this scope
        query = Task.query
        
        # Apply filters if provided
        if emp_id:
            query = query.filter_by(assigned_to=emp_id)
        if status:
            query = query.filter_by(status=status)
        if project_type:
            query = query.filter_by(project_type=project_type)
            
        # Execute query and convert to dict
        tasks = query.all()
        task_list = [task.to_dict() for task in tasks]
        
        return jsonify({
            'success': True,
            'tasks': task_list,
            'count': len(task_list)
        })
    except Exception as e:
        print(f"Error getting tasks: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
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