from flask import Flask, request
import os
import subprocess

app = Flask(__name__)

@app.route('/execute')
def execute_command():
    cmd = request.args.get('cmd', 'ls')
    output = subprocess.check_output(cmd, shell=True)
    return output.decode()

@app.route('/health')
def health_check():
    
    return "OK"

if __name__ == '__main__':
    db_password = "super_secret"
    
    # Running on all ports
    app.run(host='0.0.0.0', port=8000, debug=True)