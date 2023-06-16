import os
import subprocess
import streamlit as st

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
        subprocess.run(['python', detectorScript, '--source', filepath, '--weights', weights, '--conf', '0.25', '--name', 'detect', '--exist-ok', '--project', cacheAbsolutePath, "--no-trace"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the subprocess: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    st.title("Image Processing App")
    
    # File upload section
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    isprocessed = False

    # Image processing button
    if st.button("Process Image From custom") and uploaded_file is not None:
        # saving this file to a temporary location
        saveDir = os.path.join(os.getcwd(), 'cache')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        fileExtension=uploaded_file.name.split(".")[-1]
        fileName = "1."+fileExtension
        savePath = os.path.join(saveDir, fileName)

        with open(savePath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        run_image_processing(fileName)
        output_filepath=os.path.join(os.getcwd(),"cache/detect",fileName)
        st.image(output_filepath, caption="Processed Image")
        st.success("Image processing completed!")
    
    if st.button("Process From Coco Dataset") and uploaded_file is not None:
        # saving this file to a temporary location
        saveDir = os.path.join(os.getcwd(), 'cache')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        fileExtension=uploaded_file.name.split(".")[-1]
        fileName = "1."+fileExtension
        savePath = os.path.join(saveDir, fileName)

        with open(savePath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        run_image_processing(fileName,coco=True)
        isprocessed = True
        output_filepath=os.path.join(os.getcwd(),"cache/detect",fileName)
        st.image(output_filepath, caption="Processed Image")
        st.success("Image processing completed!")


    # Display the uploaded image
    if uploaded_file is not None and not isprocessed:
        st.subheader("Uploaded Image")
        st.image(uploaded_file)

if __name__ == "__main__":
    main()
