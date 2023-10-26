import subprocess
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cctv')
def runcode():
    subprocess.run(['python', r'C:\\project\\CCTV\\YoloDetectTest.py'], capture_output=True, text=True)
    if result.returncode != 0:
        # 스크립트 실행 중 에러 발생
        return f"Error: {result.stderr}"
    return result.stdout

if __name__ == '__main__':
    app.run(debug=True)

