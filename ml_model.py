import os
from PIL import Image
import torch

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh

import skimage.measure

from pyntcloud import PyntCloud
import matplotlib.colors
import plotly.graph_objs as go

import trimesh

import gradio as gr

state = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_state(s):
    print(s)
    global state
    state = s

def get_state():
    return state

set_state('Creating txt2mesh model...')
t2m_name = 'base40M-textvec'
t2m_model = model_from_config(MODEL_CONFIGS[t2m_name], device)
t2m_model.eval()
base_diffusion_t2m = diffusion_from_config(DIFFUSION_CONFIGS[t2m_name])

set_state('Downloading txt2mesh checkpoint...')
t2m_model.load_state_dict(load_checkpoint(t2m_name, device))


def load_img2mesh_model(model_name):
    set_state(f'Creating img2mesh model {model_name}...')
    i2m_name = model_name
    i2m_model = model_from_config(MODEL_CONFIGS[i2m_name], device)
    i2m_model.eval()
    base_diffusion_i2m = diffusion_from_config(DIFFUSION_CONFIGS[i2m_name])

    set_state(f'Downloading img2mesh checkpoint {model_name}...')
    i2m_model.load_state_dict(load_checkpoint(i2m_name, device))

    # Verify model loading
    print(i2m_model)  # Print the model object
    print(i2m_model.state_dict().keys())

    return i2m_model, base_diffusion_i2m

img2mesh_model_name = 'base40M' #'base300M' #'base1B'
i2m_model, base_diffusion_i2m = load_img2mesh_model(img2mesh_model_name)


set_state('Creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

set_state('Downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

set_state('Creating SDF model...')
sdf_name = 'sdf'
sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
sdf_model.eval()

set_state('Loading SDF model...')
sdf_model.load_state_dict(load_checkpoint(sdf_name, device))

stable_diffusion = gr.Blocks.load(name="spaces/runwayml/stable-diffusion-v1-5")


set_state('')

def get_sampler(model_name, txt2obj, guidance_scale):

    global img2mesh_model_name
    global base_diffusion_i2m
    global i2m_model
    if model_name != img2mesh_model_name:
        img2mesh_model_name = model_name
        i2m_model, base_diffusion_i2m = load_img2mesh_model(model_name)

    return PointCloudSampler(
            device=device,
            models=[t2m_model if txt2obj else i2m_model, upsampler_model],
            diffusions=[base_diffusion_t2m if txt2obj else base_diffusion_i2m, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[guidance_scale, 0.0 if txt2obj else guidance_scale],
            model_kwargs_key_filter=('texts', '') if txt2obj else ("*",)
        )

def generate_txt2img(prompt):

    prompt = f"“a 3d rendering of {prompt}, full view, white background"
    gallery_dir = stable_diffusion(prompt, fn_index=2)
    imgs = [os.path.join(gallery_dir, img) for img in os.listdir(gallery_dir) if os.path.splitext(img)[1] == '.jpg']

    return imgs[0], gr.update(visible=True)

def generate_3D(input, model_name='base40M', guidance_scale=3.0, grid_size=32):
    try:
        set_state('Entered generate function...')

        if isinstance(input, Image.Image):
            input = prepare_img(input)

        # if input is a string, it's a text prompt
        sampler = get_sampler(model_name, txt2obj=True if isinstance(input, str) else False, guidance_scale=guidance_scale)

        # Produce a sample from the model.
        print("Sampling...")
        samples = None
        kw_args = dict(texts=[input]) if isinstance(input, str) else dict(images=[input])
        for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=kw_args):
            samples = x

        set_state('Converting to point cloud...')
        pc = sampler.output_to_point_clouds(samples)[0]

        print("Saving point cloud...")
        file_dir = os.getcwd()
        pc_file_path = os.path.join(file_dir, "point_cloud.ply")
        with open(pc_file_path, "wb") as f:
            pc.write_ply(f)
        print("Point cloud saved.")

        set_state('Converting to mesh...')
        mesh_file_path = os.path.join(file_dir, "mesh.ply")
        save_ply(pc, mesh_file_path, grid_size)
        print("Mesh converted.")

        set_state('')
        
        return (
            pc_to_plot(pc),
            ply_to_obj(mesh_file_path, '3d_model.obj'),
            gr.update(value=['3d_model.obj', 'mesh.ply', 'point_cloud.ply'], visible=True)
        )

    except Exception as e:
        print("An error occurred:", str(e))
        return None  # Return an appropriate response or handle the error condition

def prepare_img(img):

    w, h = img.size
    if w > h:
        img = img.crop((w - h) / 2, 0, w - (w - h) / 2, h)
    else:
        img = img.crop((0, (h - w) / 2, w, h - (h - w) / 2))

    # resize to 256x256
    img = img.resize((256, 256))
    
    return img

def pc_to_plot(pc):

    return go.Figure(
        data=[
            go.Scatter3d(
                x=pc.coords[:,0], y=pc.coords[:,1], z=pc.coords[:,2], 
                mode='markers',
                marker=dict(
                  size=2,
                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(pc.channels["R"], pc.channels["G"], pc.channels["B"])],
              )
            )
        ],
        layout=dict(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
        ),
    )

def ply_to_obj(ply_file, obj_file):
    mesh = trimesh.load(ply_file)
    mesh.export(obj_file)

    return obj_file

def save_ply(pc, file_name, grid_size):

    # Produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=grid_size, # increase to 128 for resolution used in evals
        fill_vertex_channels=True,
        progress=True,
    )

    # Write the mesh to a PLY file to import into some other program.
    with open(file_name, 'wb') as f:
        mesh.write_ply(f)


# with gr.Blocks() as app:
#     gr.Markdown("## CAD-GPT")
        
#     with gr.Row():
#         with gr.Column():
#             with gr.Tab("Text to 3D"):
#                 prompt = gr.Textbox(label="Prompt", placeholder="A cactus in a pot")
#                 btn_generate_txt2obj = gr.Button(value="Generate")
            
#             with gr.Tab("Image to 3D"):
#                 img = gr.Image(label="Image")
#                 gr.Markdown("Best results with images of 3D objects with no shadows on a white background.")
#                 btn_generate_img2obj = gr.Button(value="Generate")

#             with gr.Tab("Text to Image to 3D"):
#                 gr.Markdown("Generate an image with Stable Diffusion, then convert it to 3D. Just enter the object you want to generate.")
#                 prompt_sd = gr.Textbox(label="Prompt", placeholder="a 3d rendering of [your prompt], full view, white background")
#                 btn_generate_txt2sd = gr.Button(value="Generate image")
#                 img_sd = gr.Image(label="Image")
#                 btn_generate_sd2obj = gr.Button(value="Convert to 3D", visible=False)

#             with gr.Accordion("Advanced settings", open=False):
#                 dropdown_models = gr.Dropdown(label="Model", value="base40M", choices=["base40M", "base300M"]) #, "base1B"])
#                 guidance_scale = gr.Slider(label="Guidance scale", value=3.0, minimum=3.0, maximum=10.0, step=0.1)
#                 grid_size = gr.Slider(label="Grid size (for .obj 3D model)", value=32, minimum=16, maximum=128, step=16)

#         with gr.Column():
#             plot = gr.Plot(label="Point cloud")
#             # btn_pc_to_obj = gr.Button(value="Convert to OBJ", visible=False)
#             model_3d = gr.Model3D(value=None)
#             file_out = gr.File(label="Files", visible=False)
            
#             # state_info = state_info = gr.Textbox(label="State", show_label=False).style(container=False)


#         # inputs = [dropdown_models, prompt, img, guidance_scale, grid_size]
#         outputs = [plot, model_3d, file_out]

#         prompt.submit(generate_3D, inputs=[prompt, dropdown_models, guidance_scale, grid_size], outputs=outputs)
#         btn_generate_txt2obj.click(generate_3D, inputs=[prompt, dropdown_models, guidance_scale, grid_size], outputs=outputs, api_name="generate_txt2obj")
        
#         btn_generate_img2obj.click(generate_3D, inputs=[img, dropdown_models, guidance_scale, grid_size], outputs=outputs, api_name="generate_img2obj")

#         prompt_sd.submit(generate_txt2img, inputs=prompt_sd, outputs=[img_sd, btn_generate_sd2obj])
#         btn_generate_txt2sd.click(generate_txt2img, inputs=prompt_sd, outputs=[img_sd, btn_generate_sd2obj], queue=False)
#         btn_generate_sd2obj.click(generate_3D, inputs=[img, dropdown_models, guidance_scale, grid_size], outputs=outputs)

#         # btn_pc_to_obj.click(ply_to_obj, inputs=plot, outputs=[model_3d, file_out])

#     '''gr.Examples(
#         examples=[
#             ["a black coffe mug"],
#             ["a cactus in a pot"],
#             ["a round table with floral tablecloth"],
#             ["a red kettle"],
#             ["a vase with flowers"],
#             ["a sports car"],
#             ["a man"],
#         ],
#         inputs=[prompt],
#         outputs=outputs,
#         fn=generate_3D,
#         cache_examples=True
#     )'''

#     '''gr.Examples(
#         examples=[
#             ["images/chair.png"],
#         ],
#         inputs=[img],
#         outputs=outputs,
#         fn=generate_3D,
#         cache_examples=True
#     )'''

#     # app.load(get_state, inputs=[], outputs=state_info, every=0.5, show_progress=False)

    
# app.queue(max_size=250, concurrency_count=6).launch(share=True)