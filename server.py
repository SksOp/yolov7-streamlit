from flask import Flask, request, send_file
import subprocess
import os
import shutil

from flask_cors import CORS,cross_origin


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000","*"]}})
@app.route('/predictCustom', methods=['POST'])
@cross_origin()  
def predictCustom():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    cache = 'cache'
    weights="api/best.pt"
    detectorScript ="api/detect.py"
    
    if not os.path.exists(cache):
        os.makedirs(cache)
    filepath = os.path.join(cache, file.filename)
    file.save(filepath)
    try:
        subprocess.run(['python', detectorScript, '--source', filepath, '--weights', weights, '--conf', '0.25', '--name', 'detect','--exist-ok','--project',cache,"--no-trace"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the subprocess: {e}")
    # You can also log the error or handle it in a different way if needed
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    saved_dir = os.path.join("../",cache, 'detect')
    output_filepath =  os.path.join(saved_dir, file.filename) # replace this with actual path and filename

    finalImage= send_file(output_filepath, mimetype='image/gif')
    # cache_dir = cache
    # for filename in os.listdir(cache_dir):
    #     file_path = os.path.join(cache_dir, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    return finalImage
@app.route('/predictCoco', methods=['POST'])
@cross_origin()  
def predictCoco():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    filepath = os.path.join('cache', file.filename)
    file.save(filepath)
    try:
        subprocess.run(['python', 'detect.py', '--source', filepath, '--weights', 'yolov7_training.pt', '--conf', '0.25', '--name', 'detect','--exist-ok','--project','cache',"--no-trace"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the subprocess: {e}")
    # You can also log the error or handle it in a different way if needed
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    output_filepath =  os.path.join('cache/detect/', file.filename) # replace this with actual path and filename

    finalImage = send_file(output_filepath, mimetype='image/gif')
    cache_dir = 'cache'
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    return finalImage

@app.after_request
def cleanup(response):
    cache_dir = 'cache'
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return response


# curl -X POST -F "file=@image.jpg" http://localhost:5000/predict > output.png
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


