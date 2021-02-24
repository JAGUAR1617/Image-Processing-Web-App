import numpy as np 
import cv2 
import streamlit as st
from PIL import Image
import altair as alt
import base64
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
st.header("Web App for Image Processing")


st.markdown("""
<style>
body {
    color: #404040;
    background-color: #ffa500  ;
}
</style>
    """, unsafe_allow_html=True)


max_lowThreshold = 100
low_threshold = 10
ratio = 3
kernel_size = 3


def threshold():
    st.header("Thresholding")
    uploaded_file = st.file_uploader("Choose a image file type .jpg", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        image = opencv_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (0, 0), fx = 0.2, fy = 0.2)
        

        
        st.write(image.shape)
        threshold_slide = st.sidebar.slider('change threshold here', min_value=10, max_value=255)
        ret, thresh1 = cv2.threshold(image,threshold_slide,255,cv2.THRESH_BINARY)
        thresh1 = thresh1.astype(np.float64)
        st.image(thresh1, use_column_width=True,clamp = True)
        st.text("Bar Chart of the image")
        histr = cv2.calcHist([image],[0],None,[256],[0,256])
        st.area_chart(histr)

         

def canny():
    st.header("Canny Edge Detection ")
    uploaded_file = st.file_uploader("Choose any jpg file for contour", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        image = opencv_image     
        st.write(image.shape)
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        threshold_canny = st.sidebar.slider('Change Value to increase or decrease threshold',min_value = 10,max_value = 255) 
        
        edges = cv2.Canny(im, threshold_canny, kernel_size)
        mask = edges != 0
        dst = im * (mask[:,:,None].astype(image.dtype))

        st.image(edges,use_column_width=True,clamp=True)
        st.image(dst,use_column_width=True,clamp=True)

def contours():
    st.header("Contours of image")
    uploaded_file = st.file_uploader("Choose any jpg file for contour", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        img = opencv_image              
        contours_slide = st.sidebar.slider('Change Value to increase or decrease threshold',min_value = 10,max_value = 255) 
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret,thresh = cv2.threshold(imgray, contours_slide, 255,0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
 
        
        st.image(thresh, use_column_width=True, clamp = True)
        st.image(img, use_column_width=True, clamp = True)


def blur():

    st.header("Blur of image")
    uploaded_file = st.file_uploader("Choose any jpg file to  blur", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        image = opencv_image          
    # blue_min = st.sidebar.slider('Change Value to increase or decrease threshold',min_value = 10,max_value = 255) 
    # blue_max = st.sidebar.slider('Change Value to increase or decrease threshold',min_value = 10,max_value = 255) 
        R = st.slider('R: Red', min_value=0, max_value=255)
        G = st.slider('G: Green', min_value=0, max_value=255)
        B = st.slider('B: Blue', min_value=0, max_value=255)

        blue_min = np.array([R, G, B])
        blue_max = np.array([255, 255, 255])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, blue_min, blue_max)
        res = cv2.bitwise_and(image, image, mask=mask)
        kernel = np.ones((3, 3), np.float32) / 225
        smoothed = cv2.filter2D(res, -1, kernel)
        bilateral = cv2.bilateralFilter(res, 15, 75, 75)
        gauss = cv2.GaussianBlur(res,(15,15),0.2)
        median = cv2.medianBlur(res, 15)
    
        st.image(median, use_column_width=True)
        st.image(bilateral, use_column_width=True)
        st.image(image, use_column_width=True)
        st.image(smoothed, use_column_width=True)
        st.image(gauss, use_column_width=True)

def Morphological_Transformations():
    st.header(" Morphological Transformations of image")
    uploaded_file = st.file_uploader("Choose any jpg file for morphological transformations", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('See Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        image = opencv_image   
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        R = st.slider('R: Red', min_value=0, max_value=255)
        G = st.slider('G: Green', min_value=0, max_value=255)
        B = st.slider('B: Blue', min_value=0, max_value=255)

        blue_min = np.array([R, G, B])
        blue_max = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, blue_min, blue_max)
        res = cv2.bitwise_and(image, image, mask=mask)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        st.image(image)
        st.image(mask)
        st.image(opening)
        st.image(closing)
 

def edge():
    st.header(" Morphological Transformations of image")
    uploaded_file = st.file_uploader("Choose any jpg file to detect corner", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('See Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        image = opencv_image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        R = st.slider('R: Red', min_value=0, max_value=255)
        G = st.slider('G: Green', min_value=0, max_value=255)
        B = st.slider('B: Blue', min_value=0, max_value=255)

        blue_min = np.array([R, G, B])   
        blue_max = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, blue_min, blue_max)
        res = cv2.bitwise_and(image, image, mask=mask)

        Laplacian = cv2.Laplacian(image, cv2.CV_8U)
        Sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
        Sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=3)

        st.image(image)
        st.code('a. orignal image')
        st.image( mask)
        st.code('b. mask image', )
        st.image(Laplacian)
        st.code('c. Laplacian image' )
        st.image( Sobelx)
        st.code('d. sobelX image' )
        st.image( Sobely)
        st.code('e. SobelY image' )

def corner_detection():
    st.header(" corner of image")
    uploaded_file = st.file_uploader("Choose any jpg file to detect corner", type="jpg")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if st.button('See Original Image'):
        
            st.image(opencv_image, use_column_width=True)
        # Now do something with the image! For example, let's display it:
        # st.image(opencv_image, channels="BGR")
        image = opencv_image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        points = st.slider('number of points around corner', min_value=10, max_value=1000)

        corners = cv2.goodFeaturesToTrack(gray, points, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 5, (255, 0, 0), -2)
        st.image(image)


def main():
    slider = st.sidebar.selectbox('Choose one of the following', ('Threshold', 'Canny', 'Contours', 'Blur', 'Morphological Transformations', 'Edges', 'Corner Detection'))

    if slider == 'Threshold':
        threshold()
    if slider == 'Canny':
        canny()
    if slider == 'Contours':
        contours()
    if slider == 'Blur':
        blur()
    if slider == 'Morphological Transformations':
        Morphological_Transformations()
    if slider == 'Edges':
        edge()
    if slider == 'Corner Detection':
        corner_detection()

if __name__ == "__main__":
    main()


st.text("Developed by : " + "Dr. Panchanand Jha")
st.text("Advanced Robotics Lab")
st.text("REC, Visakhapatnam")
st.write(" For basic follow OpenCV Book : " + "2. Jha P, Biswal B. Opencv With Python: A Basic Approach. 1st ed. 979-8686832466: KDP Amazon; 2020:74.")

