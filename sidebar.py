import streamlit as st
st.set_page_config(page_title="CryptoKnights")
with st.sidebar:
    st.image("https://i.ibb.co/8mT3Qhq/cryptoknights-removebg-preview.png")
    st.title('Medical Assistant')
    selected_model = st.sidebar.selectbox('Choose a model', ['Pneumonia', 'Brain Tumor','Search'], key='selected_model')
    st.write("<b>DISCLAIMER</b>", unsafe_allow_html=True)
    st.write("The AI system serves as a supportive tool for radiologists and healthcare practitioners. The final interpretation of medical images and the diagnosis of diseases should be made by a licensed and trained medical professional.</b>", unsafe_allow_html=True)
    show_mission = st.button("Our Mission")
    if show_mission:
        st.write("Our mission is to harness the power of artificial intelligence to improve healthcare outcomes, enhance patient care, and assist healthcare professionals in making more accurate and timely diagnoses.")
    show_future_scope = st.button("Future Scope")
    if show_future_scope:
        st.write("Improved Accuracy and Efficiency\nEarly Disease Detection\nExpanding Disease Types\nIntegration with Healthcare Systems\nContinuous Learning and Improvement\nEthical and Regulatory Considerations")
    show_contact_us = st.button("Contact Us")
    if show_contact_us:
        st.write("<b>Address:</b> Manipal, India, 576104", unsafe_allow_html=True)
        st.write("<b>Phone:</b> +91-123456789", unsafe_allow_html=True)
        