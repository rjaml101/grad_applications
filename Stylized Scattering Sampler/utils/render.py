import mitsuba as mi
mi_variant = "cuda_ad_rgb" if "cuda_ad_rgb" in mi.variants() else "llvm_rgb"
mi.set_variant(mi_variant)

import os, time
import matplotlib.pyplot as plt
import numpy as np

# Load scene
def load_scene(scene_name):
    scene_path = os.path.join("scenes", scene_name+".xml")
    return mi.load_file(scene_path)

# Time elapsed during given function operation
def time_elapsed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time Elapsed: {end_time - start_time}")
        return result
    return wrapper

# Render scene and track rendering time (only immediately trigged when SymbolicFlags==False)
@ time_elapsed
def render_scene(scene, spp=0):
    image = mi.render(scene, spp=spp)
    return image

# Export to output images
def save_output_image(output_subdir, scene_name, image):
    output_dir = os.path.join("output_images", output_subdir)
    output_filepath = os.path.join(output_dir, scene_name)
    print(output_filepath)
    print("---------------------------\n")
    mi.util.write_bitmap(output_filepath + ".png", image)
    mi.util.write_bitmap(output_filepath + ".exr", image)


