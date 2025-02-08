import os, sys
os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print("경로는:" ,os.path.dirname(os.path.abspath(os.path.dirname(__file__))))