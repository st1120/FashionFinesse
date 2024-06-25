from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import User
from . import db
import json,os,random
from random import shuffle

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt
import webcolors

def load_images_from_folder(folder_path):
    images = []
    files=[]
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
            files.append(filename)
    return images,files

def is_color_match(color, target_color, tolerance):
    return np.all(np.abs(color - target_color) < tolerance)    

def filter_images_by_color(images, target_color, tolerance, color):
    matched_images = []
    indices=[]
    
    for index, image in enumerate(images):
        if color=="white":
            center_pixel = image[image.shape[0] // 3, image.shape[1] // 2]
        else:    
            center_pixel = image[image.shape[0] // 2, image.shape[1] // 2]
        if is_color_match(center_pixel, target_color, tolerance):
            matched_images.append(image)
            indices.append(index)
    return matched_images, indices

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def load_resnet_model():
    model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
    model.trainable = False
    model = tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])

    return model

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)
       

@views.route('/submit', methods=['POST'])
@login_required
def submit():

    if request.method == 'POST': 
        event = request.form.get('events')
        gender = request.form.get('gender')
        color = request.form.get('color')

        if gender=="gender0" and event=="event0" and color=="color0": 
            flash('Please select the options', category='error')
            return render_template("home.html", user=current_user)
        elif event=="event0":
            flash('Please select an event', category='error')
            return render_template("home.html", user=current_user)
        elif gender=="gender0": 
            flash('Please select a gender', category='error')
            return render_template("home.html", user=current_user)
        elif color=="color0": 
            flash('Please select a color', category='error')
            return render_template("home.html", user=current_user)
        else:
            flash('Searching...', category='success')
 
            image_names = []         
            folder_path1 = 'website\\static\\dataset\\'+event+'\\'+gender
            folder_path2 = 'website\\static\\dataset\\'+event+'\\'+gender+'\\Accessories'
            folder_path3 = 'website\\static\\dataset\\'+event+'\\'+gender+'\\Footwear'

            images,files = load_images_from_folder(folder_path1)
            if color=="red":
                bgr=[32,20,180]
                tolerance=50
            elif color=="blue":
                bgr=[180,10,30]    
                tolerance=80
            elif color=="green":
                bgr=[20,160,30]
                tolerance=70
            elif color=="yellow":
                bgr=[22,241,250]
                tolerance=80   
            elif color=="black":
                bgr=[0,0,0]
                tolerance=60
            elif color=="white":
                bgr=[255,255,255]
                tolerance=70  
            elif color=="pink":
                bgr=[187,154,255]
                tolerance=50           
            target_color = np.array(bgr)
            filtered_images, indices = filter_images_by_color(images, target_color,tolerance,color)
            print(f"Found {len(filtered_images)} images with a center pixel matching the specified color.")
            filtered_images = filtered_images[:5]
            indices=indices[:5]
            for image_name in indices:
                image_names.append(files[image_name])
            print(image_names)
            
            acc_names = [f for f in os.listdir(folder_path2) if f.endswith(('.jpg', '.png', '.jpeg','.webp'))]
            shuffle(acc_names)
            acc_names = acc_names[:5]

            
            foot_names = [f for f in os.listdir(folder_path3) if f.endswith(('.jpg', '.png', '.jpeg','.webp'))]
            shuffle(foot_names)
            foot_names = foot_names[:5]

            
            return render_template("home.html", user=current_user,foot_names=foot_names,acc_names=acc_names, image_names=image_names,event=event,gender=gender)
            






@views.route('/image_click/<event>/<gender>/<image_path>', methods=['POST'])
@login_required



def image_click(image_path,event,gender):
    
    
    if request.method == 'POST':
        event=event
        gender=gender

        resnet_model = load_resnet_model()

        folder_path = 'website/static/dataset/'+event+'/'+gender
        newpath = 'website/static/dataset/'+event+'/'+gender+'/'+image_path



        filenames = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                filenames.append(full_path)



        feature_list = []
        for file in tqdm(filenames):
            feature_list.append(extract_features(file,resnet_model))
        
        

        img = image.load_img(newpath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = resnet_model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        

        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        
        distances, indices = neighbors.kneighbors([normalized_result])

        top_5_indices = indices.flatten()[1:7]
        top_5_image_paths = []

        for index in top_5_indices:
            str1 = filenames[index]
            filenames[index]=str1[14:]
            filenames[index]=filenames[index].replace("\\","/")
            top_5_image_paths.append(filenames[index])

        return render_template("results.html", user=current_user,top_5_image_paths=top_5_image_paths, event=event, gender=gender)    


@views.route('/accessories/<event>/<gender>', methods=['POST'])
@login_required

def accessories(event,gender):
    if request.method == 'POST':
        # if event=="Wedding" and gender=="Men":
        #     acc_path='website/static/dataset/'+event+'/'+gender+'/men-traditional/Accessories'
            
        # elif event=="Wedding" and gender=="Women":
        #     acc_path='website/static/dataset/'+event+'/'+gender+'/churidar/Accessories'

        # else:
        acc_path='website/static/dataset/'+event+'/'+gender+'/Accessories'
            
        acc_images=[f for f in os.listdir(acc_path) if f.endswith(('.jpg', '.png', '.jpeg','.webp'))]
        shuffle(acc_images)
        acc_images = acc_images[:12]

        return render_template("accessories.html", user=current_user,acc_images=acc_images, event=event, gender=gender)    


@views.route('/footwear/<event>/<gender>', methods=['POST'])
@login_required

def footwear(event,gender):
    if request.method == 'POST':

        # if event=="Wedding" and gender=="Men":
        #     foot_path='website/static/dataset/'+event+'/'+gender+'/men-traditional/Footwear'

        # elif event=="Wedding" and gender=="Women":
        #     foot_path='website/static/dataset/'+event+'/'+gender+'/churidar/Footwear'

        # else:
        foot_path='website/static/dataset/'+event+'/'+gender+'/Footwear'
        
        foot_images=[f for f in os.listdir(foot_path) if f.endswith(('.jpg', '.png', '.jpeg','.webp'))]
        shuffle(foot_images)
        foot_images = foot_images[:12]

        return render_template("footwear.html", user=current_user,foot_images=foot_images, event=event, gender=gender)    
