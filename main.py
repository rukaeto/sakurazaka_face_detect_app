#ライブラリのインポート
import streamlit as st
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import io
import os
from os.path import join, dirname
import settings
from dotenv import load_dotenv
import tensorflow as tf
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
#from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

#タイトル
st.title('櫻坂46メンバー顔認識アプリ')

#サイドバー
st.sidebar.title('さっそく顔認識をする')
st.sidebar.write('①画像をアップロード')
st.sidebar.write('②識別結果が右に表示されます。')
st.sidebar.write('--------------')
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください。", type=['jpg','jpeg', 'png'])

#Face APIの各種設定
subscription_key = settings.FACE_KEY# AzureのAPIキー
endpoint = settings.FACE_URL # AzureのAPIエンドポイント

#クライアントの認証
face_client = FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))

#メンバーリスト
members = ['上村莉菜', '尾関梨香', '小池美波', '小林由依', '齋藤冬優花', '菅井友香', '土生瑞穂',
            '原田葵', '守屋茜', '渡辺梨加', '渡邉理佐', '井上梨名', '遠藤光莉', '大園玲', '大沼晶保',
            '幸阪茉里乃', '関有美子', '武元唯衣', '田村保乃', '藤吉夏鈴', '増本綺良',
            '松田里奈', '森田ひかる', '守屋麗奈', '山﨑天']


# 各関数の定義
# モデルを読み込む関数
@st.cache
def model_load():
    model = tf.keras.models.load_model('my_model.h5')
    return model

# 顔の位置を囲む長方形の座標を取得する関数
def get_rectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    return ((left, top), (right, bottom))

# 画像に書き込むテキスト内容を取得する関数
def get_draw_text(faceDictionary):
    rect = faceDictionary.face_rectangle
    text = first[0] + ' / ' + str(round(first[1]*100,1)) + '%'
    # 枠に合わせてフォントサイズを調整
    font_size = max(30, int(rect.width / len(text)))
    font = ImageFont.truetype('SourcehanSans-VF.ttf', font_size)
    return (text, font)

# テキストを描く位置を取得する関数
def get_text_rectangle(faceDictionary, text, font):
    rect = faceDictionary.face_rectangle
    text_width, text_height = font.getsize(text)
    left = rect.left + rect.width / 2 - text_width / 2
    top = rect.top - text_height - 1
    return (left, top)

# 画像にテキストを描画する関数
def draw_text(faceDictionary):
    text, font = get_draw_text(faceDictionary)
    text_rect = get_text_rectangle(faceDictionary, text, font)
    draw.text(text_rect, text, align='center', font=font, fill='red')

# 顔部分だけの画像を作る関数
def make_face_image(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    image = np.asarray(img)
    face_image = image[top:bottom, left:right]
    cv2_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(cv2_img)
    face_image_color = cv2.merge([r,g,b])
    resized_image = cv2.resize(face_image_color, (128, 128))
    return resized_image

# 顔画像が誰なのか予測値を上位3人まで返す関数
def predict_name(image):
    img = image.reshape(1, 128, 128, 3)
    img = img / 255
    model = model_load()
    pred = model.predict(img)[0]
    top = 3
    top_indices = pred.argsort()[-top:][::-1]
    result = [(members[i], pred[i]) for i in top_indices]
    return result[0], result[1], result[2]


#以下ファイルがアップロードされた時の処理
if uploaded_file is not None:
    progress_message = st.empty()
    progress_message.write('顔を識別中です。お待ちください。')

    img = Image.open(uploaded_file)
    stream = io.BytesIO(uploaded_file.getvalue())

    detected_faces = face_client.face.detect_with_stream(stream)
    if not detected_faces:
        raise Warning('画像から顔を検出できませんでした。')

    img = Image.open(uploaded_file)
    draw = ImageDraw.Draw(img)

    face_img_list = []

    first_name_list = []
    second_name_list = []
    third_name_list = []

    first_rate_list = []
    second_rate_list = []
    third_rate_list = []

    for face in detected_faces:
        face_img = make_face_image(face)
        face_img_list.append(face_img)
        first, second, third = predict_name(face_img)
        first_name_list.append(first[0])
        first_rate_list.append(first[1])

        second_name_list.append(second[0])
        second_rate_list.append(second[1])

        third_name_list.append(third[0])
        third_rate_list.append(third[1])

        draw.rectangle(get_rectangle(face), outline='red', width=5)
        draw_text(face)

    img_array = np.array(img)
    st.image(img_array, use_column_width=True)
    col1, col2 = st.beta_columns(2)
    with col1:
        for i in range(0, len(face_img_list)):
            st.header(f'{i+1}人目:{first_name_list[i]}')
            st.image(face_img_list[i], width = 128)

    with col2:
        st.header('分析結果詳細')
        for i in range(0, len(face_img_list)):
            with st.beta_expander(f'{i+1}人目の詳細を表示'):
                st.write(first_name_list[i], 'の可能性:' , round(first_rate_list[i]*100,2), '%')
                st.write(second_name_list[i], 'の可能性:' , round(second_rate_list[i]*100,2), '%')
                st.write(third_name_list[i], 'の可能性:' , round(third_rate_list[i]*100,2), '%')

    progress_message.write(f'{len(face_img_list)}人の顔を検出しました!')
