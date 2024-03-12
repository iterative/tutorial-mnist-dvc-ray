"""
Original: https://docs.ray.io/en/latest/ray-overview/getting-started.html 
"""

import argparse
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import yaml

import ray
from ray.runtime_env import RuntimeEnv
from ray.train import ScalingConfig, RunConfig, SyncConfig
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
import torch
from torch import nn

from src.helpers import get_dataloaders
from src.model import ConvNet
from src.live import download_from_s3, list_objects_in_s3_dir


def train_func_per_worker(config: Dict):
        
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    worker_rank = ray.train.get_context().get_world_rank()

    # Get dataloaders inside worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model the correct GPU/CPU device
    # ============================================================
    model = ConvNet()
    model = ray.train.torch.prepare_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # [3] Set up Live object for DVCLive
    # ===============================
    live = None
    if worker_rank == 0:

        # Initialize DVC Live    
        from src.live import DVCLiveRayLogger as Live
        live = Live(
            dir="results/dvclive",
            save_dvc_exp=True, 
            bucket_name = "cse-cloud-version",
            s3_directory = "tutorial-mnist-dvc-ray/dvclive",  
            dvcyaml=None
        )

        print("#############################################")
        print("RAY_TRAIN_CONTEXT")
        print(ray.train.get_context().get_experiment_name())
        print(ray.train.get_context().get_metadata())
        print(ray.train.get_context().get_storage())
        print("TRIAL_DIR", ray.train.get_context().get_trial_dir())
        print("TRIAL_NAME", ray.train.get_context().get_trial_name())
        print("Live.dir", live.dir)
        print("Live.params_file", live.params_file)
        print("DVC_ENV_VARS: ", {k: v for k, v in os.environ.items() if  k.startswith("DVC")})
        print("#############################################")

    for epoch in range(epochs):

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)

                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total

        # [4] Report metrics to Ray Train
        # ===============================
        checkpoint_dir = "."
        checkpoint_path = checkpoint_dir + "/model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        ray.train.report(
            metrics={"loss": test_loss, "accuracy": accuracy},
            checkpoint=Checkpoint.from_directory(checkpoint_dir)
        )

        # [5] Log metrics with DVCLive 
        # ===============================
        if live:
            print("-------------------")
            print(f"LOG METRICS WITH DVCLIVE")
            print("worker global rank - ", ray.train.get_context().get_world_rank())
            print("worker local rank - ", ray.train.get_context().get_local_rank())
            print("node rank - ", ray.train.get_context().get_node_rank())
            print("epoch - ", epoch)
            print("loss - ", test_loss)
            print("accuracy - ", accuracy)
            print("-------------------")
            live.log_metric("loss", test_loss)
            live.log_metric("accuracy", accuracy)
            live.next_step()
            


def train(params: dict) -> None:

    # [1] Define configuration for Training
    # =============================================
    train_params = params.get("train", {})
    NUM_WORKERS = train_params.get("num_workers", 1)
    USE_GPU = train_params.get("use_gpu", False)
    GLOBAL_BATCH_SIZE = train_params.get("global_GLOBAL_batch_size", 32)
    EPOCH_SIZE = train_params.get("epoch_size", 5) 
    TRAIN_RESULTS_DIR = train_params.get("results_dir")

    TUNE_RESULTS_DIR = params.get("tune", {}).get("results_dir", "")
    BEST_PARAMS_PATH = Path(TUNE_RESULTS_DIR) / "best_params.yaml"
    BEST_MODEL_PARAMS = yaml.safe_load(open(BEST_PARAMS_PATH))

    train_config = {
        "lr": BEST_MODEL_PARAMS.get("lr", 1e-2),
        "momentum": BEST_MODEL_PARAMS.get("momentum", 0.5),
        "epochs": EPOCH_SIZE,
        "batch_size_per_worker": GLOBAL_BATCH_SIZE // NUM_WORKERS,
    }

    # [2] Configure computation resources
    # =============================================
    scaling_config = ScalingConfig(num_workers=NUM_WORKERS, use_gpu=USE_GPU)

    # [3] Runtime configuration for training and tuning runs.
    # Using cloud storage for a multi-node cluster
    # =============================================
    run_config = RunConfig(
            name="ray-trials",
            storage_path="s3://cse-cloud-version/tutorial-mnist-dvc-ray/",
            sync_config=SyncConfig(
                sync_artifacts=True, 
                sync_artifacts_on_checkpoint=True
            ),
        )

    # [3] Initialize a Ray TorchTrainer
    # =============================================
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # [4] Start Distributed Training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result: train.Result  = trainer.fit()
    print(f"Training result: {result}")
    print(f"result.filesystem: {result.filesystem}")
    print(f"result.path: {result.path}")
    print(f"result.metrics: {result.metrics}")
    print(f"result.metrics_dataframe: {result.metrics_dataframe}")
    print(f"result.config: {result.config}")
    if result.checkpoint:
        print(f"result.checkpoint.path: {result.checkpoint.path}")
    
    # [5] Save Trial Results
    # Pull Ray Trial results from S3 shared storage ("ray-trials") 
    # =============================================
    s3_path_parts = result.path.split("/", 1)
    bucket_name = s3_path_parts[0]
    results_s3_directory = s3_path_parts[1]
    
    obj_keys = list_objects_in_s3_dir(bucket_name, results_s3_directory)
    print("\nObjects in Trial S3 folder: ", obj_keys)

    for filename in ['model.pth']:
        try:
            object_key = os.path.join(results_s3_directory, filename)
            file_path = os.path.join(TRAIN_RESULTS_DIR, filename)
            download_from_s3(bucket_name, object_key, file_path)
            print(f"Downloaded {filename} from S3")
        except Exception as e:
            print(f"Error downloading {filename} from S3: {e}")

    # [6] Pull DVCLive logs from S3
    # =============================================
    s3_directory = "tutorial-mnist-dvc-ray/dvclive"
    download_from_s3(bucket_name, s3_directory, 'results/dvclive/')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch MNIST Tune Example")
    parser.add_argument("--config", help="DVC parameters")
    args, _ = parser.parse_known_args()

    # [1] Propogate DVC environment variables from Head Node to Workers 
    # =============================================
    DVC_ENV_VARS = {k: v for k, v in os.environ.items() if  k.startswith("DVC")}
    ray.init(runtime_env=RuntimeEnv(env_vars=DVC_ENV_VARS))
    
    # [2] Load config
    # =============================================
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)

    # [3] Start Training
    # =============================================
    train(params)
