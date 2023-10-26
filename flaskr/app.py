from flask import Flask, render_template, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def create_plot():
    connection = cx_Oracle.connect(user='hr', password='hr', dsn='localhost:1521/xe')
    cursor = connection.cursor()
    cursor.execute("SELECT age, COUNT(*) FROM detected_faces GROUP BY age")
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ages, counts)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Number of detected faces by age')
    
    # x축 정렬
    plt.xticks(ages, rotation=45)

    # 그래프를 이미지 파일로 저장하지 않고 BytesIO 객체에 저장합니다.
    output = BytesIO()
    FigureCanvas(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
