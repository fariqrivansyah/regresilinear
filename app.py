from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io, base64

app = Flask(__name__)

# Dataset Advertising
data = pd.read_csv("advertising.csv")  # pastikan file ada di folder yang sama

# Model Regresi Linear
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
model = LinearRegression()
model.fit(X, y)

def create_graph(tv, radio, newspaper, y_pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.tight_layout(pad=4.0)

    # TV vs Sales
    axes[0].scatter(data['TV'], y, color='blue', label='Data Aktual')
    axes[0].scatter(tv, y_pred, color='red', label='Prediksi', s=100)
    axes[0].set_xlabel("TV")
    axes[0].set_ylabel("Sales")
    axes[0].set_title("TV vs Sales")
    axes[0].legend()

    # Radio vs Sales
    axes[1].scatter(data['Radio'], y, color='green', label='Data Aktual')
    axes[1].scatter(radio, y_pred, color='red', label='Prediksi', s=100)
    axes[1].set_xlabel("Radio")
    axes[1].set_ylabel("Sales")
    axes[1].set_title("Radio vs Sales")
    axes[1].legend()

    # Newspaper vs Sales
    axes[2].scatter(data['Newspaper'], y, color='purple', label='Data Aktual')
    axes[2].scatter(newspaper, y_pred, color='red', label='Prediksi', s=100)
    axes[2].set_xlabel("Newspaper")
    axes[2].set_ylabel("Sales")
    axes[2].set_title("Newspaper vs Sales")
    axes[2].legend()

    # Simpan ke buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return f"data:image/png;base64,{graph_url}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graph_url = None

    if request.method == 'POST':
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        # Prediksi sales
        prediction = model.predict([[tv, radio, newspaper]])[0]
        graph_url = create_graph(tv, radio, newspaper, prediction)

    return render_template('index.html', prediction=prediction, graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)
