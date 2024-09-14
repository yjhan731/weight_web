import flask
from flask import Flask, request, jsonify
from gevent import monkey, pywsgi
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import numpy as np
import joblib
import pandas as pd
import os
import csv  # 导入 CSV 库

# import sys
# sys.path.append(r'E:\web\web\webserver')
# sys.path
#os.getcwd()
# 加载模型和标量器
wl_scaler = joblib.load('./wl12_scaler.pkl')
wlb_model = joblib.load('./wlb_model.pkl')
wl4_model2 = joblib.load('./wl4_model.pkl')
wl8_model2 = joblib.load('./wl8_model.pkl')
wl12_model2 = joblib.load('./wl12_model.pkl')


app = Flask(__name__)
CORS(app)



@app.route('/submit', methods=['POST'])
def submit():
    data = request.json  # 获取POST请求中的JSON数据
    print(data)
    print('precision model start!')
    if data['weight4'] == '':
        data['weight4'], data['weight8'], data['weight12'] = 60, 60, 60
        print(data)
    # 处理数据
        cate = {'time': float(data['zq'])}
        cont = {'bmi': float(data['bmi']), 'rbc': float(data['rbc']),'hgb': float(data['hgb']),
                'hdl': float(data['hdl']), 'alt': float(data['alt']), 'weightb': float(data['weightb']),
                'weight4': float(data['weight4']), 'weight8': float(data['weight8']), 'weight12':float(data['weight12'])}
        model = wlb_model
        std_fea = np.array(
            [cont['bmi'], cont['rbc'], cont['hgb'], cont['alt'], cont['hdl'], cont['weightb'], cont['weight4'],
             cont['weight8'], cont['weight12']])
        std_fea_reshape = std_fea.reshape(1, -1)
        cont_scaled = wl_scaler.transform(std_fea_reshape)
        cont_sca = list(cont_scaled[0][:6])
        cont_sca.append(cate['time'])
        input_data = np.array(cont_sca)
        print('base:', input_data)
        input_data = input_data.reshape(1, -1)
        y_pred_proba1 = model.predict(input_data)
        result = round(y_pred_proba1[0], 2)

    elif data['weight8'] == '':
        data['weight8'], data['weight12'] = 60, 60
    # 处理数据
        cate = {'time': float(data['zq'])}
        cont = {'bmi': float(data['bmi']), 'rbc': float(data['rbc']),'hgb': float(data['hgb']),
                'hdl': float(data['hdl']), 'alt': float(data['alt']), 'weightb': float(data['weightb']),
                'weight4': float(data['weight4']), 'weight8': float(data['weight8']), 'weight12':float(data['weight12'])}
        model = wl4_model2
        std_fea = np.array(
            [cont['bmi'], cont['rbc'], cont['hgb'], cont['alt'], cont['hdl'], cont['weightb'], cont['weight4'],
             cont['weight8'], cont['weight12']])
        std_fea_reshape = std_fea.reshape(1, -1)
        cont_scaled = wl_scaler.transform(std_fea_reshape)
        cont_sca = list(cont_scaled[0][:7])
        cont_sca.append(cate['time'])
        input_data = np.array(cont_sca)
        print('base+4w:', input_data)
        input_data = input_data.reshape(1, -1)
        y_pred_proba1 = model.predict(input_data)
        result = round(y_pred_proba1[0], 2)

    elif data['weight12'] == '':
        data['weight12'] = 60
    # 处理数据
        cate = {'time': float(data['zq'])}
        cont = {'bmi': float(data['bmi']), 'rbc': float(data['rbc']),'hgb': float(data['hgb']),
                'hdl': float(data['hdl']), 'alt': float(data['alt']), 'weightb': float(data['weightb']),
                'weight4': float(data['weight4']), 'weight8': float(data['weight8']), 'weight12':float(data['weight12'])}
        model = wl8_model2
        std_fea = np.array(
            [cont['bmi'], cont['rbc'], cont['hgb'], cont['alt'], cont['hdl'], cont['weightb'], cont['weight4'],
             cont['weight8'], cont['weight12']])
        std_fea_reshape = std_fea.reshape(1, -1)
        cont_scaled = wl_scaler.transform(std_fea_reshape)
        cont_sca = list(cont_scaled[0][:8])
        cont_sca.append(cate['time'])
        input_data = np.array(cont_sca)
        print('base+8w:', input_data)
        input_data = input_data.reshape(1, -1)
        y_pred_proba1 = model.predict(input_data)
        result = round(y_pred_proba1[0], 2)

    else:
        cate = {'time': float(data['zq'])}
        cont = {'bmi': float(data['bmi']), 'rbc': float(data['rbc']), 'hgb': float(data['hgb']),
                'hdl': float(data['hdl']), 'alt': float(data['alt']), 'weightb': float(data['weightb']),
                'weight4': float(data['weight4']), 'weight8': float(data['weight8']),
                'weight12': float(data['weight12'])}
        model = wl12_model2
        std_fea = np.array(
            [cont['bmi'], cont['rbc'], cont['hgb'], cont['alt'], cont['hdl'], cont['weightb'], cont['weight4'],
             cont['weight8'], cont['weight12']])
        std_fea_reshape = std_fea.reshape(1, -1)
        cont_scaled = wl_scaler.transform(std_fea_reshape)
        cont_sca = list(cont_scaled[0])
        cont_sca.append(cate['time'])
        input_data = np.array(cont_sca)
        print('base+12w:', input_data)
        input_data = input_data.reshape(1, -1)
        y_pred_proba1 = model.predict(input_data)
        result = round(y_pred_proba1[0], 2)

    response = {'message': result}
    #print("Received data:", wlb_model)
    return jsonify(response)



if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)  # 在调试模式下运行，方便调试
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()
