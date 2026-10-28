import sys
import subprocess

print("Python version:")
print(sys.version)
print("\nPackage versions:")
subprocess.run([sys.executable, "-m", "pip", "freeze"])