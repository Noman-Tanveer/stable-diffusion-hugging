from diffusers import StableDiffusionPipeline
from PIL import Image

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipe.to("cuda")

# guidance_scale 7-8.5

prompt = "a document containing tables"

image = pipe(prompt)
img = image.images[0]

# you can save the image with
img.save(f"doc.png")



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_images = 3
prompt = ["a document containing tables"] * num_images
images = pipe(prompt).images
grid = image_grid(images, rows=1, cols=3)
# you can save the grid with
grid.save(f"docs.png")
