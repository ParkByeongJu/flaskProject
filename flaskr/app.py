from flask import Flask, url_for, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key="secret_key"

ID = "hello"
PW = "world"

@app.route('/')
def home():
    if "userID" in session:
        return render_template("home.html", userid=session.get("userID"), login=True)
    else:
        return render_template("home.html", login=False)

@app.route("/login", methods=["get"])
def login():
    global ID, PW
    _id_ = request.args.get("loginId")
    _password_ = request.args.get("loginPw")

    if ID == _id_ and _password_ == PW:
        session["userId"] = _id_
        print(session["userId"])
        return redirect(url_for("home"))
    else:
        return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.pop("userID")
    return redirect(url_for("home"))

if __name__ == '__main__':
    app.run(port=5000, debug=True)