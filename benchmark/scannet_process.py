import os, sys
from tqdm.auto import tqdm

sys.path.append('./ScanNet/SensReader/python')
from SensorData import SensorData

def main(output_dir, input_dir: str = "./datasets/ScanNet/scans"):
    scenes = os.listdir(input_dir)

    for scene in tqdm(scenes):
        scene_base_dir = os.path.join(input_dir, scene)

        sd = SensorData(os.path.join(scene_base_dir, f"{scene}.sens"))

        os.makedirs(os.path.join(output_dir, scene, "depth"), exist_ok=True)
        sd.export_depth_images(os.path.join(output_dir, scene, "depth"))

        os.makedirs(os.path.join(output_dir, scene, "color"), exist_ok=True)
        sd.export_color_images(os.path.join(output_dir, scene, "color"))

        os.makedirs(os.path.join(output_dir, scene, "pose"), exist_ok=True)
        sd.export_poses(os.path.join(output_dir, scene, "pose"))

        os.makedirs(os.path.join(output_dir, scene, "intrinsic"), exist_ok=True)
        sd.export_intrinsics(os.path.join(output_dir, scene, "intrinsic"))

        del sd

if __name__ == '__main__':
    import fire
    fire.Fire(main)
