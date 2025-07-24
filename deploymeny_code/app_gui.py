
import streamlit as st
import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import cv2

IMG_W = 256
IMG_H = 256
MODEL_WEIGHTS_PATH = 'deploymeny_code/best_model_layers_weights'

def build_generator_v2(img_w, img_h):
    def SE_Block(input_tensor, reduction=16):
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(filters // reduction, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape((1, 1, filters))(se)
        return Multiply()([input_tensor, se])

    def ResidualBlock(x, filters):
        shortcut = x
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Add()([shortcut, x])
        x = LeakyReLU(0.2)(x)
        return x

    inputs = Input((img_h, img_w, 2))

    e1 = Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    e1 = LeakyReLU(0.2)(e1)
    e2 = Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(e1)
    e2 = BatchNormalization(momentum=0.9)(e2)
    e2 = LeakyReLU(0.2)(e2)
    e3 = Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal')(e2)
    e3 = BatchNormalization(momentum=0.9)(e3)
    e3 = LeakyReLU(0.2)(e3)
    e4 = Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal')(e3)
    e4 = BatchNormalization(momentum=0.9)(e4)
    e4 = LeakyReLU(0.2)(e4)
    b = Conv2D(1024, 4, strides=2, padding='same', kernel_initializer='he_normal')(e4)
    b = BatchNormalization(momentum=0.9)(b)
    b = Activation('relu')(b)

    for _ in range(2):
        b = ResidualBlock(b, 1024)

    b = SE_Block(b)

    d1 = Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal')(b)
    d1 = BatchNormalization(momentum=0.9)(d1)
    d1 = Dropout(0.2)(d1)
    d1 = Activation('relu')(d1)
    d1 = Concatenate()([d1, e4])
    d2 = Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer='he_normal')(d1)
    d2 = BatchNormalization(momentum=0.9)(d2)
    d2 = Dropout(0.2)(d2)
    d2 = Activation('relu')(d2)
    d2 = Concatenate()([d2, e3])
    d3 = Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(d2)
    d3 = BatchNormalization(momentum=0.9)(d3)
    d3 = Activation('relu')(d3)
    d3 = Concatenate()([d3, e2])
    d4 = Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(d3)
    d4 = BatchNormalization(momentum=0.9)(d4)
    d4 = Activation('relu')(d4)
    d4 = Concatenate()([d4, e1])
    d5 = Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer='he_normal')(d4)
    d5 = BatchNormalization(momentum=0.9)(d5)
    d5 = Activation('relu')(d5)

    structure = Conv2D(3, 3, padding='same', activation='tanh', kernel_initializer='glorot_normal')(d5)
    color_refine = Conv2D(3, 1, padding='same', activation='tanh', kernel_initializer='zeros')(d5)
    outputs = Add()([0.8 * structure, 0.2 * color_refine])

    return Model(inputs, outputs)

def load_layer_weights(model, save_dir='layer_weights'):
    for i, layer in enumerate(model.layers):
        layer_path = os.path.join(save_dir, f'layer_{i}.npz')
        if os.path.exists(layer_path):
            data = np.load(layer_path)
            weights = [data[f'arr_{j}'] for j in range(len(data.files))]
            layer.set_weights(weights)

def preprocess_image(image):
    edge_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.resize(edge_img, (IMG_W, IMG_H))
    edge_img = (edge_img / 255.0) * 2 - 1
    edge_img = edge_img.astype(np.float32)
    edge_img = np.squeeze(edge_img)
    noise_channel = np.random.uniform(-1, 1, size=edge_img.shape).astype(np.float32)
    return np.stack([edge_img, noise_channel], axis=-1)

def predict_image(model, img_with_noise):
    input_img = np.expand_dims(img_with_noise, axis=0)
    output_img = model.predict(input_img)[0]
    output_img = (output_img + 1) / 2
    return (output_img * 255).astype(np.uint8)

st.title("Edge-to-Image Generator")
uploaded_file = st.file_uploader("Upload Edge Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    input_tensor = preprocess_image(image)
    model = build_generator_v2(IMG_W, IMG_H)
    load_layer_weights(model, MODEL_WEIGHTS_PATH)
    output = predict_image(model, input_tensor)

    cols = st.columns(2)
    cols[0].image(image, caption="Input Image", channels="BGR", use_container_width =True)
    cols[1].image(output, caption="Generated Image", channels="RGB", use_container_width =True)
