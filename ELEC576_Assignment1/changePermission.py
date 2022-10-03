import os

folder = os.getcwd() + "\\results"
print(folder)
os.chmod(folder, 0o777)
for file in os.listdir(folder):
    os.chmod(file, 0o777)
os.chmod(folder + "/val", 0o777)
for file in os.listdir(folder + "/val"):
    os.chmod(file, 0o777)
os.chmod(folder + "/test", 0o777)
for file in os.listdir(folder + "/test"):
    os.chmod(file, 0o777)


