import submitit
from submitit.helpers import Checkpointable, DelayedSubmission

import os
from enum import Enum
from typing import Optional

NYU_PATH = "REPLACE_ME" # path to the NYU Depth V2 dataset


class SlurmJobType(Enum):
    CPU = 0
    GPU = 1


def is_slurm_available() -> bool:
    return submitit.AutoExecutor(".").cluster == "slurm"


def setup_slurm(
    name: str,
    job_type: SlurmJobType,
    submitit_folder: str = "submitit",
    depend_on: Optional[str] = None,
    timeout: int = 180,
    high_compute_memory: bool = False,
) -> submitit.AutoExecutor:
    os.makedirs(submitit_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=submitit_folder, slurm_max_num_timeout=10)

    ################################################
    ##                                            ##
    ##   ADAPT THESE PARAMETERS TO YOUR CLUSTER   ##
    ##                                            ##
    ################################################

    # You may choose low-priority partitions where job preemption is enabled as
    # any preempted jobs will automatically resume/restart when rescheduled.

    if job_type == SlurmJobType.CPU:
        kwargs = {
            "slurm_partition": "compute",
            "gpus_per_node": 0,
            "slurm_cpus_per_task": 14,
            "slurm_mem": "32GB" if not high_compute_memory else "64GB",
        }
    elif job_type == SlurmJobType.GPU:
        kwargs = {
            "slurm_partition": "low-prio-gpu",
            "gpus_per_node": 1,
            "slurm_cpus_per_task": 4,
            "slurm_mem": "32GB",
            # If your cluster supports choosing specific GPUs based on constraints,
            # you can uncomment this line to select low-memory GPUs.
            "slurm_constraint": "p40",
        }

    ###################
    ##               ##
    ##   ALL DONE!   ##
    ##               ##
    ###################

    kwargs = {
        **kwargs,
        "slurm_job_name": name,
        "timeout_min": timeout,
        "tasks_per_node": 1,
        "slurm_additional_parameters": {"depend": f"afterany:{depend_on}"}
        if depend_on is not None
        else {},
    }

    executor.update_parameters(**kwargs)

    return executor

def get_marigold_model():
    import sys
    sys.path.append("PATH_TO_MARIGOLD_REPOSITORY_CLONE")

    from marigold import MarigoldPipeline

    marigold = MarigoldPipeline.from_pretrained("PATH_TO_MARIGOLD_CHECKPOINT")
    try:
        import xformers
        marigold.enable_xformers_memory_efficient_attention()
    except:
        pass
    return marigold

def run_inference_for_category(category_id, out_path):
    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm
    from torch.utils.data import Dataset, DataLoader

    class CategoryDataset(Dataset):
        def __init__(self, category_id, out_path):
            self.category_id = category_id

            self.category_path = os.path.join(NYU_PATH, str(category_id))

            if os.path.exists(os.path.join(out_path, str(category_id))):
                images_processed = len(os.listdir(os.path.join(out_path, str(category_id))))
            else:
                images_processed = 0

            print(f"Found {images_processed} images that have already been processed")

            self.images = sorted(os.listdir(self.category_path))[images_processed:]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image_name = self.images[idx]
            image_path = os.path.join(self.category_path, image_name)

            image = Image.open(image_path).convert("RGB")

            return image_name, image

    print(f"This runner is for category {category_id}")

    os.makedirs(os.path.join(out_path, category_id), exist_ok=True)

    dataset = CategoryDataset(category_id, out_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    marigold = get_marigold_model().to("cuda")

    print("Initialized Marigold model")

    for image_names in tqdm(dataloader):
        image_name, image = image_names[0]

        w, h = image.size

        marigold_out = marigold(
            image.crop((8, 8, 632, 472)),
            denoising_steps=10,
            ensemble_size=10,
            match_input_res=True,
            batch_size=0,
            color_map="Spectral",
            show_progress_bar=False,
        )

        out_image_arr = marigold_out["depth_np"].squeeze()

        # pad back to original size
        out_image_arr = np.pad(out_image_arr, ((8, 8), (8, 8)), mode="constant", constant_values=0)

        np.save(os.path.join(out_path, str(category_id), image_name.replace(".jpg", ".npy")), out_image_arr)

class CategoryInference(Checkpointable):
    def __call__(self, *args, **kwargs):
        return run_inference_for_category(*args, **kwargs)

    def checkpoint(self, *args, **kwargs) -> DelayedSubmission:
        """Resubmits the same callable with the same arguments"""
        return DelayedSubmission(self, *args, **kwargs)  # type: ignore

def run_inference_for_all_categories(out_path):
    os.makedirs(out_path, exist_ok=True)

    category_ids = sorted(os.listdir(NYU_PATH))

    if is_slurm_available():
        print("SLURM is available")

        executor = setup_slurm(
                    f"nyudepth",
                    SlurmJobType.GPU,
                    timeout=48 * 60,
                )

        executor.update_parameters(slurm_array_parallelism=20)

        with executor.batch():
            for category_id in category_ids:
                executor.submit(CategoryInference(), category_id, out_path)

        print(f"Submitted {len(category_ids)} jobs to SLURM")

    else:
        from tqdm.auto import tqdm
        for category_id in tqdm(category_ids):
            run_inference_for_category(category_id, out_path)

    
def main(out_path):
    run_inference_for_all_categories(out_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

# %%
