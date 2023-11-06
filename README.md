# 🎓 Tutorial: Automated Distributed ML Pipelines with DVC and Ray
This tutorial will guide users through creating automated, scalable, and distributed ML pipelines using DVC (Data Version Control) integrated with Ray. 

- Run Distributed ML Pipeline with DVC [DVC](https://dvc.org/). 
- Design distributed ML pipeliens with Ray [Ray](https://www.ray.io/). 
- Introduce [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) for hyperparameter optimization 
- Run distributed ML experiment with DVC and Ray on AWS 

![DVC + Ray](src/static/preview.png "DVC + Ray")

# Install 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Ray Cluster (local)

Locally: cpu, single-node

```bash
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --head --port=6379
```

Run DVC pipeline 

```bash
dvc exp run
```

### Examples 

Submit Ray job 
```bash
ray job submit --working-dir . -- python examples/scalable_dataset.py
ray job submit --working-dir . -- python examples/tune_get_started.py
ray job submit --working-dir . -- python examples/pytorch_mnist.py
```

## Run Cluster in Docker

```bash
docker run --shm-size=8G -v .://home/ray/ray-dvc \
    -p 8265:8265 \
    -t -i rayproject/ray-ml:nightly-py310-cpu
```

Submit Ray job
```bash
ray job submit --working-dir . -- python ray-dvc/examples/scalable_dataset.py
```

## Stop Ray Cluster 
```bash
ray stop
```


# TODO 
- Add `save_dvclive` stage 
- Add `results/experiment_name` to `.gitignore`
- Test with Studio 
- Test with AWS cluster