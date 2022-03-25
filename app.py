import streamlit as st

import requests
from PIL import Image
from src.predict import *
from src.streamlit_utils import load_stuff


# Load the models
model, preprocess, extractor, nouns_features, device = load_stuff()

pydiffvg.set_print_timing(False)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)

# Load the model
# model, preprocess = clip.load('ViT-B/32', device, jit=False)


prompt = st.text_input("Enter a prompt:")

image_path = st.text_input('Style image URL', value='https://i.pinimg.com/originals/5d/e5/1d/5de51d8b1f795cb5728ad5b43d238a04.jpg')
uploaded_img = st.file_uploader("Upload style image...", type=['jpg', 'png', 'jpeg'])

if uploaded_img:
    style_image = Image.open(uploaded_img).convert('RGB')
else:
    style_image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')


st.image(style_image, 'style image')

num_paths = st.slider('Number of Paths', 1, 1000, 250)
num_iterations = st.slider('Number of iterations', 1, 2000, 250)
style_strength = st.slider('Style strength', 1, 100, 50)
black_and_white = st.checkbox('Black and white', False)
uniform_width = st.checkbox('Uniform width', False)
slow = st.checkbox('Use Slow Version (More Style Influence)', False)


if st.button('Run!'):
    assert isinstance(num_paths, int) and num_paths > 0, 'num_paths should be an positive integer'
    assert isinstance(num_iterations, int) and num_iterations > 0, 'num_iterations should be an positive integer'
    assert isinstance(style_strength, int) and 0 <= style_strength <= 100, \
        'style_strength should be a positive integer less than 100'
    assert style_image is not None, 'style_image must be specified'
    assert prompt is not None and len(prompt) > 0, 'prompt must be specified'

    style_weight = 4 * (style_strength / 100)

    paths = []
    shape_groups_out = []

    pbar = st.progress(0)
    in_progress_img = st.empty()

    if slow:
        for path, shape_groups in style_clip_draw_slow(prompt, style_image, model, extractor, num_paths=num_paths,
                                                  num_iter=num_iterations, style_weight=style_weight, num_augs=4,
                                                  st_pbar=pbar,):
            paths.append(path), shape_groups_out.append(shape_groups)
            in_progress_img.image(str(path), 'in progress')

    else:
        for path, shape_groups in style_clip_draw(prompt, style_image, model, extractor, num_paths=num_paths,
                                    num_iter=num_iterations, style_weight=style_weight, num_augs=10, st_pbar=pbar,
                                    black_and_white=black_and_white, uniform_width=uniform_width):
            paths.append(path), shape_groups_out.append(shape_groups)
            in_progress_img.image(str(path), 'in progress')

    st.image(str(paths[-1]), 'final image')

    with tempfile.NamedTemporaryFile() as f:
        pydiffvg.save_svg(f.name, **shape_groups_out[-1])
        st.download_button('Download SVG', f.read(), 'out_svg.svg')
