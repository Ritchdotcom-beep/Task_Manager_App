import pickle

with open('task_assignment_model.plk', 'rb') as f:
    data = pickle.load(f)
print(data)
