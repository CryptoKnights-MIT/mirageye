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
from google_trans_new import google_translator 
from pygoogletranslation import Translator
import googletrans

from gtts import gTTS
from io import BytesIO

#Global variable
wiki_api_url = "https://en.wikipedia.org/w/api.php"

Languages = {'afrikaans':'af','albanian':'sq','amharic':'am','arabic':'ar','armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn','bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr','czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi','french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht','hausa':'ha','hawaiian':'haw','hebrew':'iw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is','igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km','korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb','macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn','myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt','punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd','sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw','swedish':'sv','tajik':'tg','tamil':'ta','telugu':'te','thai':'th','turkish':'tr','turkmen':'tk','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy','xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}

# Global functions
def feedback():
    st.header("Feedback Form")
    feedback = st.text_area("Please provide your feedback:")

    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
    else:
        st.warning("Please enter your feedback before submitting.")


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
    plt.title("Using Gradcam")
    return plt
#api rel functions

dont_include="this the and over under small other about above below right nearby faraway  in out when where if location of may result on as a an from since why how or by might may will should would could they act body it "

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
def format_with_spaces(text):
    # Define a regular expression pattern to match special characters
    special_chars_pattern = r'([!@#$%^&*()_+{}\[\]:;"\'<>,.?/\|\\])'

    # Use re.sub() to replace special characters with spaces before and after
    formatted_text= re.sub(special_chars_pattern,r' \1',text)
    formatted_text=' '.join(formatted_text.split())
    return formatted_text
# App title
st.set_page_config(page_title="CryptoKnights")


# Sidebar
with st.sidebar:
    st.image("https://i.ibb.co/TTPW5R5/l6fbgue7.png")
    st.title('Medical Assistant')
    selected_model = st.sidebar.selectbox('Choose a model', ['Pneumonia', 'Brain Tumor','Search',"Feedback"], key='selected_model')
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        show_mission = st.button("Mission")
        
    with col2:
        show_future_scope = st.button("Future")
    
    with col3:
        show_contact_us = st.button("Contact")

    if show_future_scope:
        st.write("<ul><li>Improved Accuracy and Efficiency</li><li>Early Disease Detection</li><li>Expanding Disease Types</li><li>Integration with Healthcare Systems</li><li>Continuous Learning and Improvement</li><li>Ethical and Regulatory Considerations</li></ul>", unsafe_allow_html=True)

    if show_mission:
        st.write("Our mission is to harness the power of artificial intelligence to improve healthcare outcomes, enhance patient care, and assist healthcare professionals in making more accurate and timely diagnoses.")
    
    if show_contact_us:
        st.write("<ul><li><b>Address:</b> Manipal, India, 576104</li><li><b>Phone:</b> +91-123456789</li></ul>", unsafe_allow_html=True)

    st.subheader("DISCLAIMER")
    st.write("This simply serves as a supportive tool for diagnosis. The final interpretation of medical images should only be made by a licensed doctor.</b>", unsafe_allow_html=True)
    

    
    
    
# Going into the model
match selected_model:
     
    case "Pneumonia":
            
        # Upload image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="pneu")
        if uploaded_image is not None:
            st.toast("Image uploaded successfully", icon='👩‍⚕️')
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

                st.toast("Prediction complete",icon='👩‍⚕️')
                total = y_prob[0][1] + y_prob[0][2]
                st.write(f"<h4>You have a <b>{total*100:.2f}% </b> chance of having Pneumonia</h4>", unsafe_allow_html=True)
                st.write(f"<ul><li><b>Bacterial Pneumonia:</b> {y_prob[0][1]*100:.2f}% chance</li><li><b>Viral Pneumonia:</b> {y_prob[0][2]*100:.2f}% chance</li></ul>", unsafe_allow_html=True)

                # Heatmap
                st.subheader("Heatmap")
                fig = heatmapgen(image)   
                st.pyplot(fig)

                # Gradcam
                st.subheader("Severity Map")
                layer_name = 'conv2d_1'
                input_shape = (150,150,3)
                with st.spinner("Generating gradcam..."):
                    score_cam = ScoreCam(model, p_img, layer_name, input_shape)
                st.toast("Severity Map Generated", icon='👩‍⚕️')
                score_cam_superimposed = superimpose(uploaded_image, score_cam)
                plt = gengradcambox(score_cam_superimposed)
                st.pyplot(plt)
                st.markdown(f"The box is the region with highest amount of pathogens",unsafe_allow_html=True)

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
                st.write(f"<ul><li>High Severity: {percent_red:.2f}%</li><li>Medium Severity: {percent_yel:.2f}%</li><li>Low Severity: {percent_blu:.2f}%</li></ul>", unsafe_allow_html=True)

                st.toast("Please fill the feedback form", icon='👩‍⚕️')
                
            del model
                
                

    case "Brain Tumor":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="brain")
        class_names = ['Glioma Tumor', 'Meningioma Tumor', 'Normal', 'Pituitary Tumor']

        if uploaded_image is not None:
            st.toast("Image uploaded successfully", icon='👩‍⚕️')
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

                st.toast("Prediction complete", icon='👩‍⚕️')    
                st.subheader("Prediction")        
                st.write(f"<h6>{prediction_label}</h6>", unsafe_allow_html=True)

                # Heatmap
                st.subheader("Heatmap")
                plt.figure(figsize=(6, 6))
                fig, ax = plt.subplots()
                sns.heatmap(img[0, :, :, 0], cmap="viridis")
                plt.title(f"Heatmap of Input Image ({prediction_label})")
                st.pyplot(fig)

                # Gradcam
                st.subheader("Severity Map")
                layer_name = 'block5_conv3'
                input_shape = (224,224,3)
                with st.spinner("Generating Severity Map..."):
                
                    score_cam = ScoreCam(model, img, layer_name, input_shape, max_N=25)
                
                
                score_cam_superimposed = superimpose(uploaded_image, score_cam)
                st.toast("Severity Map Generated", icon='👩‍⚕️')
                plt = gengradcambox(score_cam_superimposed)
                st.pyplot(plt)
                st.markdown(f"The box signifies the tumor",unsafe_allow_html=True)

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
                st.write(f"<ul><li>High Severity: {percent_red:.2f}%</li><li>Medium Severity: {percent_yel:.2f}%</li><li>Low Severity: {percent_blu:.2f}%</li></ul>", unsafe_allow_html=True)
                st.toast("Please fill the feedback form", icon='👩‍⚕️')
                del model
                
                
    case "Search":
        translator = Translator()
        option1 = st.selectbox('Preferred language',
                      ('english', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch',  'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'))
        if(option1):
            value1 = Languages[option1]
            disease_name=st.text_input("Enter Disease name:","")
            if(disease_name):
                wiki_content = fetch_wikipedia_content(disease_name)
                if wiki_content:
                    disease_info = summarize_text(wiki_content)
                    disease_info1=format_with_spaces(disease_info).split()
                    info=[]
                    for word in disease_info1:
                        if fetch_wikipedia_content(word)  and word not in dont_include and len(word)>=5:
                            # D8C4B6
                            info.append(f"<span style='background-color:purple;'>{word}</span>")

                        else:
                            info.append(f"{word}")
                    info=" ".join(info)
                    output=f"Information about {disease_name}"
                    st.text(output)
                    summary=f"Description:"
                    if(value1 != 'en'):
                        disease_info = translator.translate(disease_info,src='en',dest=value1)
                        disease_info =  disease_info.text
                        st.markdown(disease_info)
                    else:
                        st.text(summary)
                        st.markdown(info, unsafe_allow_html=True)
                    try:
                        tts = gTTS(text=disease_info, lang=value1)
                        audio_buffer = BytesIO()
                        tts.write_to_fp(audio_buffer)

                        # Display the audio using the Audio component
                        audio_buffer.seek(0)
                        st.audio(audio_buffer, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    output=f"Failed to retrieve Wikipedia content for {disease_name}"
                    st.text(output)
                
    case "Feedback":
        feedback()          








