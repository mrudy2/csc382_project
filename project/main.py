import subprocess

print("Running Logistic Regression...")
subprocess.run(["python", "logres_rudy.py"], check=True)

print("Running SVM and Decision Tree...")
subprocess.run(["python", "project_rudy.py"], check=True)

print("All steps completed successfully.")