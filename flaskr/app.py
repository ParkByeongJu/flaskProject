from flask import Flask, url_for, render_template, request, redirect, session
import cx_Oracle as db

app = Flask(__name__)
app.secret_key="secret_key"

def db():
    dsn = db.makedsn('localhost', 1521, 'xe')
    con = db.connect("hr", "hr", dsn)
    cursor = con.cursor()
    return cursor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/LoginProcess')
def loginProcess():
    pass

if __name__ == '__main__':
    app.run(port=5000, debug=True)