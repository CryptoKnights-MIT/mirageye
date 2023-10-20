# Import CNN modules
import time
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle

from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

#Global variable
wiki_api_url = "https://en.wikipedia.org/w/api.php"

# Global functions

# Gradcam functions
def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def ScoreCam(model, img_array, layer_name, input_shape, max_N=-1):
    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]
    
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    weights = pred_from_masked_input_array[:,cls]
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam) 
    cam /= np.max(cam) 
    return cam

def superimpose(uploaded_image, cam, emphasize=False):
    image = Image.open(uploaded_image)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def read_and_preprocess_img(path, size=(224,224)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Generate a heatmap from the given unprocessed image
def heatmapgen(image):
    img_array = np.array(image)
    image_df = pd.DataFrame(img_array, columns=range(img_array.shape[1]))
    fig, ax = plt.subplots()
    sns.heatmap(image_df, cmap='YlGnBu', annot=False)
    return fig

def gengradcambox(score_cam_superimposed):
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    # Create a mask that identifies the red area
    mask = np.all((score_cam_superimposed >= lower_red) & (score_cam_superimposed <= upper_red), axis=-1)
    # Find the largest connected component
    label_image = label(mask, connectivity=2)
    regions = regionprops(label_image)
    # Find the largest component
    largest_component = max(regions, key=lambda region: region.area)
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    ax.imshow(score_cam_superimposed, cmap='jet')
    # Calculate the bounding box for the largest component
    minr, minc, maxr, maxc = largest_component.bbox
    # Draw a bounding box around the largest component
    rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, color='black', linewidth=1)
    ax.add_patch(rect)
    # Add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(0, 100)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Severity (%)", orientation='vertical')
    # Display the plot
    ax.axis('off')
    return plt
#api rel functions

dont_include="this the and in out when where if location of may result on as a an from since why how or "

def fetch_wikipedia_content(article_title):
    # Parameters for the Wikipedia API request
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": article_title,
        "exintro": True,
        "explaintext": True,
    }

    try:
        response = requests.get(wiki_api_url, params=params)
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        content = page.get("extract", "")
        return content
    except Exception as e:
        print(f"Failed to retrieve Wikipedia content: {e}")
        return None
def summarize_text(text):
    # Load T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize and summarize the text
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
def insert_newlines(text, words_per_line=10):
    # Split the text into words using whitespace as the separator
    words = text.split()
    
    # Use a list comprehension to join words into lines with a specified number of words per line
    lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    
    # Join the lines with newline characters
    result = '\n'.join(lines)
    
    return result
class TextColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    REVERSE = "\033[7m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
def format_with_spaces(text):
    # Define a regular expression pattern to match special characters
    special_chars_pattern = r'([!@#$%^&*()_+{}\[\]:;"\'<>,.?/\|\\])'

    # Use re.sub() to replace special characters with spaces before and after
    formatted_text = re.sub(special_chars_pattern, r' \1 ', text)

    # Remove extra spaces that might have been added
    formatted_text = ' '.join(formatted_text.split())

    return formatted_text


# App title
st.set_page_config(page_title="⚕️ Mirageye")


# Sidebar
with st.sidebar:
    st.title('⚕️ Mirageye')
    st.subheader('Models and Parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Pneumonia', 'Brain Tumor','Search'], key='selected_model')

# Going into the model
match selected_model:
     
    case "Pneumonia":
            
        # Upload image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="pneu")
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Load model with GPU
            with tf.device('/gpu:0'):
                model = load_model(r"/home/rohan/hackathon/cnn/pneumonia_multi/model.h5")
            #Prediction
            if st.button("Predict"):
                #Image Pre-processing
                p_img = image.convert("RGB")
                p_img = p_img.resize((150, 150))
                p_img = np.array(p_img) / 255.0
                p_img = np.expand_dims(p_img, axis=0)
                
                #Prediction loading
                with st.spinner("Predicting..."):
                    y_prob = model.predict(p_img)
                    y_pred = y_prob.argmax(axis=-1)
                    #time.sleep(2)

                st.subheader("Prediction")
                total = y_prob[0][1] + y_prob[0][2]
                st.write(f"You have a {total*100:.2f}% chance of having pneumonia.")
                st.write(f"(Bacterial: {y_prob[0][1]*100:.2f}% chance) (Viral: {y_prob[0][2]*100:.2f}% chance)")

                # Heatmap
                st.subheader("Heatmap")
                fig = heatmapgen(image)   
                st.pyplot(fig)

                # Gradcam
                st.subheader("Gradcam")
                layer_name = 'conv2d_1'
                input_shape = (150,150,3)
                with st.spinner("Generating gradcam..."):
                    score_cam = ScoreCam(model, p_img, layer_name, input_shape)
                
                score_cam_superimposed = superimpose(uploaded_image, score_cam)
                plt = gengradcambox(score_cam_superimposed)
                st.pyplot(plt)

                # Calculate percentage
                size=score_cam_superimposed.size
                RED_MIN=np.array([0,0,128],np.uint8)
                RED_MAX=np.array([250,250,255],np.uint8) 
                dstr=cv2.inRange(score_cam_superimposed,RED_MIN,RED_MAX)
                no_red=cv2.countNonZero(dstr)
                frac_red=np.divide(float(no_red),int(size))
                percent_red=np.multiply(float(frac_red),100)
                BLU_MIN=np.array([128,0,0],np.uint8)
                BLU_MAX=np.array([255,250,250],np.uint8) 
                dstr2=cv2.inRange(score_cam_superimposed,BLU_MIN,BLU_MAX)
                no_blu=cv2.countNonZero(dstr2)
                frac_blu=np.divide(float(no_blu),int(size))
                percent_blu=np.multiply(float(frac_blu),100)
                percent_yel=100-percent_red-percent_blu
                txt_percent_red = f"Percent Area in high severity: {percent_red}"
                txt_percent_blu = f"Percent Area in medium severity: {percent_yel}"
                txt_percent_yel = f"Percent Area in low severity: {percent_blu}"
                st.text(txt_percent_red)
                st.text(txt_percent_blu)
                st.text(txt_percent_yel)
                del model
                
                

    case "Brain Tumor":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="brain")
        class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Load model with GPU
            with tf.device('/gpu:0'):
                model = load_model(r"/home/rohan/hackathon/cnn/braintumor_expanded/brain_tumour.h5")
            
            # Prediction
            if st.button("Predict"):
                #Image Pre-processing
                img = load_img(uploaded_image, target_size=(224, 224))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                # Prediction loading
                with st.spinner("Predicting..."):
                    predictions = model.predict(img)
                    predicted_class = np.argmax(predictions[0])
                    prediction_label = class_names[predicted_class]

                st.subheader("Prediction")                
                st.write(prediction_label)

                # Heatmap
                st.subheader("Heatmap")
                plt.figure(figsize=(6, 6))
                fig, ax = plt.subplots()
                sns.heatmap(img[0, :, :, 0], cmap="viridis")
                plt.title(f"Heatmap of Input Image ({prediction_label})")
                st.pyplot(fig)

                # Gradcam
                st.subheader("Gradcam")
                layer_name = 'block5_conv3'
                input_shape = (224,224,3)
                with st.spinner("Generating gradcam..."):
                    score_cam = ScoreCam(model, img, layer_name, input_shape, max_N=25)
                
                score_cam_superimposed = superimpose(uploaded_image, score_cam)
                plt = gengradcambox(score_cam_superimposed)
                st.pyplot(plt)

                size=score_cam_superimposed.size
                RED_MIN=np.array([0,0,128],np.uint8)
                RED_MAX=np.array([250,250,255],np.uint8) 
                dstr=cv2.inRange(score_cam_superimposed,RED_MIN,RED_MAX)
                no_red=cv2.countNonZero(dstr)
                frac_red=np.divide(float(no_red),int(size))
                percent_red=np.multiply(float(frac_red),100)
                BLU_MIN=np.array([128,0,0],np.uint8)
                BLU_MAX=np.array([255,250,250],np.uint8) 
                dstr2=cv2.inRange(score_cam_superimposed,BLU_MIN,BLU_MAX)
                no_blu=cv2.countNonZero(dstr2)
                frac_blu=np.divide(float(no_blu),int(size))
                percent_blu=np.multiply(float(frac_blu),100)
                percent_yel=100-percent_red-percent_blu
                txt_percent_red = f"Percent Area in high severrity: {percent_red}"
                txt_percent_blu = f"Percent Area in medium severity: {percent_yel}"
                txt_percent_yel = f"Percent Area in low severity: {percent_blu}"
                st.text(txt_percent_red)
                st.text(txt_percent_blu)
                st.text(txt_percent_yel)
                del model
    case "Search":
        disease_name=st.text_input("Enter Disease name:","")
        if(disease_name):
            wiki_content = fetch_wikipedia_content(disease_name)
            if wiki_content:
                disease_info = summarize_text(wiki_content)
                disease_info=format_with_spaces(disease_info).split()
                info=""
                color_code="\033[43m"
                for word in disease_info:
                    if fetch_wikipedia_content(word)  and word not in dont_include:
                        info=info+" "+f"{color_code}{word}{TextColor.RESET}"
                    else:
                        info=info+" "+word
                output=f"Information about {disease_name}"
                st.text(output)
                summary=f"Description:{info}"
                summary=insert_newlines(summary)
                st.text(summary)
            else:
                output=f"Failed to retrieve Wikipedia content for {disease_name}"
                st.text(output)
                








