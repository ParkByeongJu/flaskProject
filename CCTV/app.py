from flask import Flask, render_template
import cx_Oracle as db
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():

    # Oracle 데이터베이스에 연결하기 위한 정보
    dsn = db.makedsn(host='localhost', port=1521, service_name='xe')
    user = 'hr'
    password = 'hr'

    # 데이터베이스 연결
    connection = db.connect(user=user, password=password, dsn=dsn)
    cursor = connection.cursor()

    # 쿼리 실행
    query = "SELECT age, COUNT(*) FROM detected_faces GROUP BY age"
    cursor.execute(query)

    # 데이터 가져오기
    data = cursor.fetchall()

    # 데이터 분리 및 나이 구간 첫 번째 숫자 기준 정렬
    data.sort(key=lambda x: int(x[0].strip('()').split(',')[0]))
    ages = [row[0] for row in data]
    counts = [row[1] for row in data]

    # 막대 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.bar(ages, counts)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Number of detected faces by age')

    # x축 정렬
    plt.xticks(ages, rotation=45)

    # 이미지를 바이트스트림으로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # 연결 종료
    cursor.close()
    connection.close()

    # HTML에 이미지를 전달
    return render_template('index2.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
