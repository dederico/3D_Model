from flask import Flask, request, send_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from ml_model import generate_3D

app = Flask(__name__)

@app.route('/generate_3d_model', methods=['POST'])
def generate_3d_model():
    # Extract the necessary data from the request
    input_data = request.json['prompt']
    model_name = request.json.get('model_name', 'base40M')
    guidance_scale = request.json.get('guidance_scale', 3.0)
    grid_size = request.json.get('grid_size', 32)

    # Generate the 3D model and create a matplotlib figure
    fig = generate_3D(input_data, model_name, guidance_scale, grid_size)

    # Save the figure as a PNG image file
    image_file = BytesIO()
    plt.savefig(image_file, format='png')
    image_file.seek(0)

    # Return the saved image file
    return send_file(image_file, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
