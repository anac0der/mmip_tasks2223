from PIL import Image

def make_gif():
    frames = []
    for i in range(1000):
        try:
            frames.append(Image.open(f"./test_iteration_{i}.png"))
        except Exception:
            pass
    frame_one = frames[0]
    frame_one.save("result_astranaut.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=200, loop=0)

make_gif()