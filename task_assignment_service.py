# task_assignment_service.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import os
import requests
import pickle
import json
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_for_task_service')

# Employee service configuration
EMPLOYEE_SERVICE_URL = os.environ.get('EMPLOYEE_SERVICE_URL', 'http://localhost:5001/api')
API_KEY = os.environ.get('API_KEY', 'dev_api_key')

# Define project types and their required skills
PROJECT_TYPES = {
    "website_development": [
        "HTML", "CSS", "JavaScript", "React", "Vue", "Angular", "Node.js", 
        "PHP", "UI/UX", "Responsive Design", "Web Security"
    ],
    "mobile_app_development": [
        "Swift", "Kotlin", "React Native", "Flutter", "Java", "Mobile UI/UX", 
        "Firebase", "App Store Optimization"
    ],
    "machine_learning": [
        "Python", "TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision", 
        "Data Mining", "Statistics", "Feature Engineering"
    ],
    "data_engineering": [
        "SQL", "ETL", "Data Warehouse", "Spark", "Hadoop", "Data Modeling", 
        "MongoDB", "PostgreSQL", "AWS Redshift"
    ],
    "api_development": [
        "REST API", "GraphQL", "Node.js", "Django", "Flask", "API Security",
        "API Testing", "API Documentation", "Microservices"
    ],
    "devops": [
        "Docker", "Kubernetes", "CI/CD", "AWS", "Azure", "GCP", "Jenkins",
        "Terraform", "Ansible", "System Administration"
    ],
    "blockchain": [
        "Solidity", "Smart Contracts", "Ethereum", "Web3.js", "DApps",
        "Blockchain Security", "Consensus Algorithms"
    ],
    "cybersecurity": [
        "Network Security", "Penetration Testing", "Vulnerability Assessment",
        "Security Auditing", "Encryption", "Ethical Hacking", "OWASP"
    ],
    "game_development": [
        "Unity", "Unreal Engine", "C#", "C++", "Game Design", "3D Modeling",
        "Animation", "Physics Simulation", "Multiplayer Networking"
    ]
}

# Helper functions for API calls to employee service
def api_headers():
    return {'X-API-KEY': API_KEY, 'Content-Type': 'application/json'}

def get_all_employees():
    """Get all employees from the employee service"""
    try:
        response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees', headers=api_headers())
        if response.status_code == 200:
            return response.json()
        print(f"Failed to get employees: {response.status_code}, {response.text}")
        return []
    except Exception as e:
        print(f"Exception while getting employees: {str(e)}")
        return []

# Function to load or train the model
def load_or_train_model():
    model_path = 'task_assignment_model.pkl'
    
    # Check if model already exists
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If loading fails, train a new model
    
    # Train a new model
    return train_model(model_path)

def train_model(model_path):
    print("Training new task assignment model...")
    
    # Get all employees
    employees_data = get_all_employees()
    if not employees_data:
        print("Error: Could not fetch employees data")
        # Return a simple default model that just assigns based on skills
        return {'model': None, 'feature_columns': []}
    
    # Ensure employees_data is a list of dictionaries
    if isinstance(employees_data, str) or not isinstance(employees_data, list):
        print(f"Error: Unexpected employees data format: {type(employees_data)}")
        # Return a simple default model
        return {'model': None, 'feature_columns': []}
    
    # Create synthetic tasks based on employee skills
    try:
        tasks = create_synthetic_tasks(employees_data)
    except Exception as e:
        print(f"Error creating synthetic tasks: {str(e)}")
        return {'model': None, 'feature_columns': []}
    
    # Feature engineering and model training
    try:
        model = build_assignment_model(employees_data, tasks)
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model
    except Exception as e:
        print(f"Error building model: {str(e)}")
        return {'model': None, 'feature_columns': []}

def create_synthetic_tasks(employees):
    """Create synthetic tasks based on employee skills"""
    synthetic_tasks = []
    task_id_counter = 1000
    
    # Extract all unique skills
    all_skills = set()
    for emp in employees:
        # Ensure emp is a dictionary
        if not isinstance(emp, dict):
            print(f"Warning: Employee is not a dictionary: {emp}")
            continue
            
        if 'skills' in emp and emp['skills']:
            # Ensure skills is a list
            if isinstance(emp['skills'], list):
                all_skills.update(emp['skills'])
            else:
                print(f"Warning: Skills is not a list: {emp['skills']}")
    
    # Create tasks for common skill combinations
    for project_type, skills in PROJECT_TYPES.items():
        # Filter to skills that exist in our employees
        relevant_skills = [skill for skill in skills if skill in all_skills]
        
        if not relevant_skills:
            continue
            
        # Create various tasks with different combinations of skills
        for i in range(5):  # Create 5 tasks per project type
            # Select 2-4 skills randomly
            num_skills = min(np.random.randint(2, 5), len(relevant_skills))
            selected_skills = np.random.choice(relevant_skills, num_skills, replace=False)
            
            task = {
                "task_id": f"TASK{task_id_counter}",
                "project_type": project_type,
                "skills": list(selected_skills),
                "complexity": np.random.choice(["Low", "Medium", "High"]),
                "priority": np.random.choice(["Low", "Medium", "High"])
            }
            
            synthetic_tasks.append(task)
            task_id_counter += 1
    
    return pd.DataFrame(synthetic_tasks)

def build_assignment_model(employees, tasks_df):
    """Build the ML model for task assignment"""
    # Process employees
    if not employees:
        return {'model': None, 'feature_columns': []}
        
    # Convert to DataFrame or ensure it's a DataFrame
    if isinstance(employees, list):
        employees_df = pd.DataFrame(employees)
    else:
        employees_df = employees
    
    # Convert skills to string lists if they're not already
    employees_df['skills'] = employees_df['skills'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    # Extract all unique skills
    all_skills = set()
    for skills in employees_df['skills']:
        if skills:
            all_skills.update(skills)
    
    # Create skill matrices for employees
    employee_skill_matrix = pd.DataFrame(0, index=employees_df.index, columns=list(all_skills))
    for idx, skills in enumerate(employees_df['skills']):
        for skill in skills:
            if skill in all_skills:  # Check if skill exists in our column set
                employee_skill_matrix.loc[idx, skill] = 1
    
    # Generate synthetic historical assignments
    historical_assignments = []
    
    for _, task in tasks_df.iterrows():
        # Find employees with matching skills
        task_skills = set(task['skills'])
        
        for idx, emp_row in employees_df.iterrows():
            emp_skills = set(emp_row['skills'])
            emp_id = emp_row['emp_id']
            
            # Calculate skill match
            skill_match = len(task_skills.intersection(emp_skills)) / len(task_skills) if task_skills else 0
            
            # Skip if no skill match
            if skill_match == 0:
                continue
                
            # Calculate experience and success rate
            experience = emp_row.get('experience', 0)
            success_rate = emp_row.get('success_rate', 0.0)
            
            # Record the assignment with features
            assignment = {
                "task_id": task['task_id'],
                "emp_id": emp_id,
                "project_type": task['project_type'],
                "complexity": 0 if task['complexity'] == "Low" else 1 if task['complexity'] == "Medium" else 2,
                "priority": 0 if task['priority'] == "Low" else 1 if task['priority'] == "Medium" else 2,
                "skill_match_percentage": skill_match * 100,
                "experience": experience,
                "success_rate": success_rate,
                "tasks_completed": emp_row.get('tasks_completed', 0)
            }
            
            historical_assignments.append(assignment)
    
    if not historical_assignments:
        print("Error: No valid assignments generated")
        return {'model': None, 'feature_columns': []}
        
    # Convert to DataFrame
    historical_df = pd.DataFrame(historical_assignments)
    
    # Prepare features and target
    feature_cols = ['complexity', 'priority', 'skill_match_percentage', 
                   'experience', 'success_rate', 'tasks_completed']
    
    X = historical_df[feature_cols]
    y = historical_df["emp_id"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'feature_columns': feature_cols
    }

def get_skills_for_project_type(project_type):
    """Get the required skills for a given project type"""
    normalized_type = project_type.lower().replace(' ', '_')
    
    if normalized_type in PROJECT_TYPES:
        return PROJECT_TYPES[normalized_type]
    
    # If the exact project type is not found, try to find a similar one
    for pt, skills in PROJECT_TYPES.items():
        if normalized_type in pt or pt in normalized_type:
            return skills
    
    # If no match, return general programming skills
    return ["Programming", "Problem Solving", "Communication"]

def assign_tasks_ml(tasks, employees, model_data):
    """Assign tasks to employees using ML model"""
    if not employees or not isinstance(employees, list):
        print(f"Error: Invalid employees data: {type(employees)}")
        return {"error": "Invalid employees data"}
        
    if not tasks or not isinstance(tasks, list):
        print(f"Error: Invalid tasks data: {type(tasks)}")
        return {"error": "Invalid tasks data"}
    
    assignments = {}
    
    # Track assigned workload
    current_workloads = defaultdict(int)
    for emp in employees:
        if isinstance(emp, dict) and 'emp_id' in emp:
            current_workloads[emp['emp_id']] = 0
    
    # Process each task
    for task in tasks:
        if not isinstance(task, dict):
            print(f"Error: Task is not a dictionary: {task}")
            assignments[str(task)] = "Invalid task format"
            continue
            
        task_id = task.get('task_id', 'unknown')
        project_type = task.get('project_type', '')
        required_skills = task.get('skills', [])
        
        # Generate candidate features for each employee
        candidates = []
        
        for emp in employees:
            if not isinstance(emp, dict) or 'emp_id' not in emp:
                continue
                
            emp_id = emp['emp_id']
            emp_skills = emp.get('skills', [])
            
            # Ensure skills is a list
            if not isinstance(emp_skills, list):
                emp_skills = []
            
            # Calculate skill match
            skill_match = 0
            if required_skills and emp_skills:
                matching_skills = set(required_skills).intersection(set(emp_skills))
                skill_match = len(matching_skills) / len(required_skills)
            
            # Skip if no skill match
            if skill_match == 0:
                continue
                
            # Create feature vector for this employee-task pair
            features = {
                "emp_id": emp_id,
                "name": emp.get('name', ''),
                "complexity": 0 if task.get('complexity', '') == "Low" else 1 if task.get('complexity', '') == "Medium" else 2,
                "priority": 0 if task.get('priority', '') == "Low" else 1 if task.get('priority', '') == "Medium" else 2,
                "skill_match_percentage": skill_match * 100,
                "experience": emp.get('experience', 0),
                "success_rate": emp.get('success_rate', 0),
                "tasks_completed": emp.get('tasks_completed', 0),
                "current_workload": current_workloads[emp_id]
            }
            
            candidates.append(features)
        
        if not candidates:
            assignments[task_id] = "No eligible employees found for this task"
            continue
        
        # Convert to DataFrame
        candidates_df = pd.DataFrame(candidates)
        
        # If we have ML model available and it has feature columns defined
        if model_data and 'model' in model_data and model_data['model'] and 'feature_columns' in model_data:
            model = model_data['model']
            feature_cols = model_data['feature_columns']
            
            # Get features for model prediction if all required columns exist
            all_cols_exist = all(col in candidates_df.columns for col in feature_cols)
            
            if all_cols_exist:
                X_candidates = candidates_df[feature_cols].copy()
                
                # Calculate custom score based on priorities
                candidates_df["custom_score"] = (
                    0.7 * candidates_df["skill_match_percentage"] / 100 +  # 70% weight for skill match
                    0.2 * candidates_df["success_rate"] / 100 +            # 20% weight for success rate
                    0.1 * candidates_df["experience"] / 10                 # 10% weight for experience
                )
                
                # Apply workload penalty for high workload employees
                candidates_df.loc[candidates_df["current_workload"] > 3, "custom_score"] *= 0.8
                
                # Try to use model prediction probabilities
                try:
                    employee_probs = model.predict_proba(X_candidates)
                    model_scores = np.max(employee_probs, axis=1)
                    
                    # Calculate final score: 70% custom score + 30% model score
                    candidates_df["final_score"] = 0.7 * candidates_df["custom_score"] + 0.3 * model_scores
                except Exception as e:
                    print(f"Model prediction error: {str(e)}")
                    candidates_df["final_score"] = candidates_df["custom_score"]
            else:
                print(f"Not all columns exist for model: {feature_cols}")
                candidates_df["final_score"] = candidates_df["skill_match_percentage"] / 100
        else:
            # Fallback scoring when no model is available
            candidates_df["final_score"] = (
                0.7 * candidates_df["skill_match_percentage"] / 100 +  # 70% weight for skill match
                0.2 * candidates_df["success_rate"] / 100 +            # 20% weight for success rate
                0.1 * candidates_df["experience"] / 10                 # 10% weight for experience
            )
        
        # Sort by final score
        candidates_df = candidates_df.sort_values("final_score", ascending=False)
        
        # Assign to the best employee
        best_emp = candidates_df.iloc[0]
        emp_id = best_emp["emp_id"]
        
        assignments[task_id] = {
            "emp_id": emp_id,
            "name": best_emp["name"],
            "skill_match_percentage": f"{best_emp['skill_match_percentage']:.1f}%",
            "score": f"{best_emp['final_score']:.3f}"
        }
        
        # Update workload
        current_workloads[emp_id] += 1
    
    return assignments

# API Routes
@app.route('/api/create_task', methods=['POST'])
def create_task():
    """Create a new task and get assignment recommendations"""
    # Note: We're removing the session check here as it should be handled by the main app
    # Task service should focus on the assignment logic
        
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
    
    # Get all employees
    employees = get_all_employees()
    
    # Load or train the model
    model_data = load_or_train_model()
    
    # Assign task using ML model
    assignments = assign_tasks_ml(tasks, employees, model_data)
    
    if isinstance(assignments, dict) and "error" in assignments:
        return jsonify({
            'success': False,
            'error': assignments["error"]
        }), 500
    
    return jsonify({
        'success': True,
        'task': task_data,
        'assignments': assignments
    })


@app.route('/api/get_project_skills', methods=['GET'])
def get_project_skills():
    """Get skills required for a specific project type"""
    project_type = request.args.get('project_type')
    if not project_type:
        return jsonify({'success': False, 'error': 'Project type is required'}), 400
        
    skills = get_skills_for_project_type(project_type)
    return jsonify({'success': True, 'skills': skills})


@app.route('/api/task-service/project-types', methods=['GET'])
def get_project_types():
    """Return all available project types and their skills"""
    return jsonify({
        'project_types': list(PROJECT_TYPES.keys()),
        'project_type_details': PROJECT_TYPES
    })

@app.route('/api/task-service/skills-for-project', methods=['GET'])
def get_skills_for_project():
    """Return the required skills for a given project type"""
    project_type = request.args.get('project_type')
    
    if not project_type:
        return jsonify({'error': 'Project type is required'}), 400
    
    skills = get_skills_for_project_type(project_type)
    return jsonify({'skills': skills})

@app.route('/api/task-service/assign-tasks', methods=['POST'])
def assign_tasks():
    """Assign tasks to employees using ML model"""
    try:
        data = request.json
        
        if not data or 'tasks' not in data:
            return jsonify({'success': False, 'error': 'Tasks are required'}), 400
        
        tasks = data['tasks']
        
        # Process tasks to ensure they have skills
        for task in tasks:
            if 'project_type' in task and ('skills' not in task or not task['skills']):
                task['skills'] = get_skills_for_project_type(task['project_type'])
        
        # Get all employees
        employees = get_all_employees()
        
        # Load or train the model
        model_data = load_or_train_model()
        
        # Assign tasks
        assignments = assign_tasks_ml(tasks, employees, model_data)
        
        if isinstance(assignments, dict) and "error" in assignments:
            return jsonify({
                'success': False,
                'error': assignments["error"]
            }), 500
        
        return jsonify({
            'success': True,
            'assignments': assignments
        })
    except Exception as e:
        import traceback
        print(f"Error in assign_tasks: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Exception: {str(e)}'
        }), 500

@app.route('/api/task-service/retrain-model', methods=['POST'])
def retrain_model():
    """Force retraining of the ML model"""
    # Delete existing model if it exists
    model_path = 'task_assignment_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Train new model
    model_data = train_model(model_path)
    
    if model_data:
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to retrain model'
        }), 500

if __name__ == '__main__':
    app.run(port=5002, debug=True)