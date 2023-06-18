import os
import subprocess
import streamlit as st
import sys
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable deprecated warning

# Set the custom plot style
# plt.style.use('dark_background')  # Set the background to dark
# plt.rcParams['axes.facecolor'] = (14/255, 17/255, 23/255)  # Set the background color using RGB values
# plt.rcParams['lines.color'] = 'white'  # Set line color
# plt.rcParams['axes.edgecolor'] = (14/255, 17/255, 23/255)   # Set axis edge color
# plt.rcParams['axes.labelcolor'] = 'white'  # Set axis label color
# plt.rcParams['xtick.color'] = 'white'  # Set x-axis tick color
# plt.rcParams['ytick.color'] = 'white'  # Set y-axis tick color

st.set_page_config(page_title="Object Detection Web App", layout="wide",page_icon="assets/icon.png",
                   
                   )

def run_image_processing(file,coco=False):
    # Define your image processing logic here


    cache = 'cache'
    weights = "yolov7_training.pt" if coco else "best.pt"
    detectorScript = "detect.py"

    cacheAbsolutePath = os.path.join(os.getcwd(), cache)
    if not os.path.exists(cacheAbsolutePath):
        os.makedirs(cacheAbsolutePath)
    detectPath = os.path.join(cacheAbsolutePath, 'detect')
    if not os.path.exists(detectPath):
        os.makedirs(detectPath)
    filepath = os.path.join(cacheAbsolutePath, file)
    try:
        python_executable = sys.executable
        subprocess.run([python_executable , detectorScript, '--source', filepath, '--weights', weights, '--conf', '0.25', '--name', 'detect', '--exist-ok', '--project', cacheAbsolutePath, "--no-trace"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the subprocess: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    return True



def resize_image(image_path, max_width):

    # Open the image using Pillow
    image = Image.open(image_path)

    # Get the original dimensions
    width, height = image.size

    # Calculate the new height while maintaining the aspect ratio
    new_width = max_width
    new_height = int(height * (new_width / width))

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(image_path)



def load_model_visualizer():
    st.subheader("Model visualization")
    st.download_button('Download file', 'best.onnx.png')
    st.caption("Download the image to see the clear picture of the model")
    st.image("best.onnx.png", caption="Model" )

def process_uploaded_image(uploaded_file,coco=False):
    saveDir = os.path.join(os.getcwd(), 'cache')

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    

    fileExtension=uploaded_file.name.split(".")[-1]
    fileName = "1."+fileExtension
    savePath = os.path.join(saveDir, fileName)

    with open(savePath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    resize_image(savePath, 1000)

    run_image_processing(fileName,coco=coco)

    output_filepath=os.path.join(os.getcwd(),"cache/detect",fileName)
    col1,col2 = st.columns(2)
    with col1:
        st.image(output_filepath, caption="Processed Image")
    with col2:
        st.image(uploaded_file, caption="Original Image")


def uploadPic(coco=False):
    col1,dumb= st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"],key=f"fileUploader{coco}")
        # Image processing button
    if st.button("Process Image",key=f"buton@fileUplaod{coco}") and uploaded_file is not None:
        # saving this file to a temporary location
        process_uploaded_image(uploaded_file,coco=coco)
def takePic(coco=False):
    takenPic = st.camera_input("Take a picture",key=f"camera{coco}")
    if takenPic is not None:
        process_uploaded_image(takenPic,coco=coco)

def load_coco_dataset():
    st.caption("You can run the model with the weights trained on the coco dataset. ")
    st.markdown("To know more about the coco dataset, click [here](https://cocodataset.org/#home)") 
    dropdownContainer , dumb ,dumb= st.columns(3)
    with dropdownContainer:
        st.subheader("Try it !")
        dropdown= st.selectbox('Uplaod / Take picture ', ['Upload', 'Take Picture'] ,key=f"dropdown for coco" )
    if dropdown == 'Upload':
        uploadPic(coco=True)
    else:
        takePic(coco=True)    

def load_custom_dataset():

    st.caption("The associated classes are: ")
    class1, class2, class3, class4, class5,dumb,dumb = st.columns(7)
    with class1:
        st.caption("1. Mobile/Tablet")
    with class2:
        st.caption("2. Laptop")
    with class3:
        st.caption("3. TV/Monitor")
    with class4:
        st.caption("4. Keyboard")
    with class5:
        st.caption("5. Mouse")
    
    dropdownContainer , dumb ,dumb= st.columns(3)
    with dropdownContainer:
        st.subheader("Try it !")
        dropdown= st.selectbox('Uplaod / Take picture ', ['Upload', 'Take Picture'],key=f"dropdown for dropdown")
    if dropdown == 'Upload':
        uploadPic()
    else:
        takePic()

    st.markdown("These weights are trained on the custom dataset. To download the weights, click [here](https://github.com/SksOp/yolov7-streamlit/blob/main/best.pt)")
    # with takepic:
    #     initiate = st.button("Take a picture")
    #     if initiate:
    #         takenPic = st.camera_input("Take a picture")



def load_info():
    st.write('''
        This is a simple web application that performs object detection on images. The app utilizes the [YOLOv7 model](https://github.com/WongKinYiu/yolov7), which has been trained on a custom dataset with specific classes.\n
    #### Classes
        

            The YOLOv7 model has been trained to detect the following custom classes:
    
            1. Mobile/Tablet
            2. Laptop
            3. TV/Monitor
            4. Keyboard
            5. Mouse
    
    ''')


    # Dataset section
    st.markdown("#### The Dataset")
    st.markdown("The dataset consists of 442 images based on the above-mentioned classes. To get the dataset, click [here](https://www.kaggle.com/datasets/sksop47/pc-setup-detector-dataset).")
    data = {
    'Class': ['Phone/Tablet', 'Laptop', 'TV/Monitor', 'Keyboard', 'Mouse'],
    'Training': [155, 229, 129, 90, 64],
        'Validation': [17, 33, 10, 7, 8]
    }
    df = pd.DataFrame(data)
    table , dumb = st.columns(2)
    with table:
        st.table(df)
    # Performance section
    st.markdown("#### Performance")
    st.markdown("##### Loss")
    col1, col2 ,dumb= st.columns(3)

    # Training loss graphs
    with col1:
        st.markdown("Train Loss")
        train_obj_loss_df = pd.read_csv("csv/train_obj_loss.csv")
        train_box_loss_df = pd.read_csv("csv/train_box_loss.csv")
        train_class_loss_df = pd.read_csv("csv/train_class_loss.csv")
        fig1, ax1 = plt.subplots()
        ax1.plot(train_obj_loss_df["Step"], train_obj_loss_df["train/obj_loss"], label="Object Loss")
        ax1.plot(train_box_loss_df["Step"], train_box_loss_df["train/box_loss"], label="Box Loss")
        ax1.plot(train_class_loss_df["Step"], train_class_loss_df["train/class_loss"], label="Class Loss", linestyle='--')
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_ylim(0, 0.15)
        st.pyplot(fig1)

    # Validation/Object Loss
    with col2:
        st.markdown("Validation Loss")
        val_obj_loss_df = pd.read_csv("csv/val_obj_loss.csv")
        val_box_loss_df = pd.read_csv("csv/val_box_loss.csv")
        val_class_loss_df = pd.read_csv("csv/val_class_loss.csv")
        fig2, ax2 = plt.subplots()
        ax2.plot(val_obj_loss_df["Step"], val_obj_loss_df["val/obj_loss"], label="Object Loss")
        ax2.plot(val_box_loss_df["Step"], val_box_loss_df["val/box_loss"], label="Box Loss")
        ax2.plot(val_class_loss_df["Step"], val_class_loss_df["val/class_loss"], label="Class Loss" ,linestyle='--' )
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.set_xlabel("Steps")
        ax2.set_ylim(0, 0.15)
        st.pyplot(fig2)

    col3, dumb,dumb = st.columns(3)
    with col3:
        st.markdown("Confusion Matrix")
        st.image("assets/cf.png", caption="Confusion Matrix")

    st.markdown("#### Sample detection while training")
    col4,col5,dumb = st.columns(3)

    with col4:

        st.image("assets/tb1.jpg", caption="train batch 5")

    with col5:

        st.image("assets/tb2.jpg", caption="train batch 9")
    st.markdown("##### To get the Whole report, click [here](https://api.wandb.ai/links/shubhaman47/s34r4k4n) ")
    st.empty()

    st.markdown("#### Sample Input and Output Images after training")
    col7, col8 = st.columns(2)

    with col7:
        st.caption("##### Sample Input Image")
        st.image("assets/smi.png")

    # # Sample output image
    with col8:
        st.caption("##### Sample Predicted Image")
        st.image("assets/smo.jpg")

def load_Details():
    st.markdown("### Github")
    st.markdown("To see the code, click [here](https://github.com/SksOp/yolov7-streamlit)")
    st.markdown("### Wandb")
    st.markdown("To see the wandb report, click [here](https://api.wandb.ai/links/shubhaman47/s34r4k4n)")

def load_heading():
    top_banner_html = """
        <div class="banner_holder">
        <img src="https://res.cloudinary.com/dbz9wkcqz/image/upload/v1687020196/Frame_1_gdyqhl.png" >
        </div>
    """
    css = os.path.join(os.getcwd(), "style.css")
    with open(css) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.markdown(top_banner_html, unsafe_allow_html=True)
    st.title("Object detection APP")


def main():

    load_heading()
    tab0, tab1, tab2, tab3 , tab4= st.tabs(["Introduction","Try on our weights", "Coco class" , "Explore Model","Details"])
    
    with tab0:
        load_info()
    with tab1:
        load_custom_dataset()
    with tab2:
        load_coco_dataset()
    with tab3:
        load_model_visualizer()
    with tab4:
        load_Details()


    # # Display the uploaded image
    # if uploaded_file is not None and not isprocessed:
    #     st.subheader("Uploaded Image")
    #     st.image(uploaded_file)

if __name__ == "__main__":
    main()
