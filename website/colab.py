# from IPython.display import display, Image
# import os
# import ipywidgets as widgets
# import random

# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# import numpy as np
# from numpy.linalg import norm
# import os
# from tqdm import tqdm


# from sklearn.neighbors import NearestNeighbors
# import cv2
# import matplotlib.pyplot as plt

# def recommend(event,gender):
#     newpath=""
#     folder_path="dataset/Beach"
#     if(gender=="male"):
#         folder_path="dataset/Beach/Men"
#     else:
#         folder_path="dataset/Beach/Women"
#     if(folder_path):
#         selected_files = []
#         for subfolder in os.listdir(folder_path):
#             subfolder_path = os.path.join(folder_path, subfolder)
#             image_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif','.webp'))]
#             if len(image_files) > 0:
#                 random_image = random.choice(image_files)
#                 selected_files.append(os.path.join(subfolder_path, random_image))

#         random.shuffle(selected_files)
#         for file in selected_files[:7]:
#             image_widget = widgets.Image(value=open(file, "rb").read(), width=100, height=100)

#             button = widgets.Button(description="Show Similar", layout=widgets.Layout(width='100px'))

#             def on_image_click(b, path=file):
#                 global newpath
#                 newpath = path
#                 global folder_of_clicked_image
#                 folder_of_clicked_image = os.path.dirname(path)
#                 print(f"Clicked on image: {path}")

#             button.on_click(on_image_click)
#             button_container.children += (widgets.VBox([image_widget, button]),)

#         display(button_container)
#     else:
#         image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif','.webp'))]

#         random.shuffle(image_files)
#         selected_files = image_files[:5]

#         for file in selected_files:
#             image_path = os.path.join(folder_path, file)
#             image_widget = widgets.Image(value=open(image_path, "rb").read(), width=100, height=100)

#             button = widgets.Button(description="Show Similar", layout=widgets.Layout(width='100px'))

#             def on_image_click(b, path=image_path):
#                 global newpath
#                 newpath = path
#                 print(f"Clicked on image: {path}")

#             button.on_click(on_image_click)
#             button_container.children += (widgets.VBox([image_widget, button]),)

#     display(button_container)

# def process():


# model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
# model.trainable = False

# model = tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])

# def extract_features(img_path,model):
#     img = image.load_img(img_path,target_size=(224,224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)

#     return normalized_result

# filenames = []
# if ((folder_path=="/content/drive/MyDrive/Dataset/Train/wedding collections/Male") or (folder_path=="/content/drive/MyDrive/Dataset/Train/wedding collections/Female")):
#     for file in os.listdir(folder_of_clicked_image):
#         filenames.append(os.path.join(folder_of_clicked_image,file))
# else:
#      for file in os.listdir(folder_path):
#         filenames.append(os.path.join(folder_path,file))

# feature_list = []

# for file in tqdm(filenames):
#     feature_list.append(extract_features(file,model))

# def final():


# img = image.load_img(newpath, target_size=(224, 224))
# img_array = image.img_to_array(img)
# expanded_img_array = np.expand_dims(img_array, axis=0)
# preprocessed_img = preprocess_input(expanded_img_array)
# result = model.predict(preprocessed_img).flatten()
# normalized_result = result / norm(result)

# neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
# neighbors.fit(feature_list)

# distances, indices = neighbors.kneighbors([normalized_result])

# plt.figure(figsize=(10, 10))
# for i, file_index in enumerate(indices[0][1:6], 1):
#     temp_img = cv2.imread(filenames[file_index])
#     temp_img_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
#     plt.subplot(1, 5, i)
#     plt.imshow(temp_img_rgb)
#     plt.axis('off')

# plt.show()





from flask import Flask, render_template, request
import os
import random
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

folder_path = os.path.join("website","dataset","Beach")

filenames = []

for file in os.listdir(folder_path):
    filenames.append(os.path.join(folder_path, file))

feature_list = []

for file in filenames:
    feature_list.append(extract_features(file, model))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    gender = request.form.get('gender')  # Add the name attribute to your gender dropdown in the HTML
    event = request.form.get('event')  # Add the name attribute to your event dropdown in the HTML

    if gender == "male":
        folder_path = os.path.join("website","dataset","Beach","Men")
    else:
        folder_path = os.path.join("website","dataset","Beach","Women")

    selected_files = []

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        image_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
        if len(image_files) > 0:
            random_image = random.choice(image_files)
            selected_files.append(os.path.join(subfolder_path, random_image))

    random.shuffle(selected_files)

    return render_template('recommend.html', images=selected_files)
