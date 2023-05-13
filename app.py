from flask import Flask, request

from ml_model import generate_3D

app = Flask(__name__)

@app.route('/generate_3d_model', methods=['POST'])
def generate_3d_model():
    # Extract the necessary data from the request
    input_data = request.json['prompt']
    model_name = request.json.get('model_name', 'base40M')
    guidance_scale = request.json.get('guidance_scale', 3.0)
    grid_size = request.json.get('grid_size', 32)

    # Generate the 3D model using the imported function from the notebook
    point_cloud, obj_file, _ = generate_3D(input_data, model_name, guidance_scale, grid_size)

    # Return the generated point cloud and obj file
    return {
        'point_cloud': point_cloud,
        'obj_file': obj_file
    }

if __name__ == '__main__':
    app.run()