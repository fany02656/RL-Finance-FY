# RL-Finance

## Setting up Singularity

### Getting the Overlay (Virtual File System)

The overlay file is used as a virtual file system by Singularity.

When you run Singularity and write any files to the `/ext3` directory, those
files will be saved in the overlay.  The advantage of this approach is that it
enables you to write many more files than are permitted by your Greene
allocation: Greene treats the entire overlay as just one file, but from within a
Singularity container, the overlay can contain millions of files.

We should all use the same overlay when using Singularity. This will guarantee
that we are all using the same Python package versions, data, etc.

My overlay file is located at:
`/scratch/ars991/rl_finance_singularity/rl_finance.ext3`

Copy that file to some place in your scratch directory, and store the path to it
in an environment variable.

```bash
export SINGULARITY_OVERLAY_PATH=/path/to/overlay/rl_finance.ext3
echo "export SINGULARITY_OVERLAY_PATH=\"${SINGULARITY_OVERLAY_PATH}\"" >> ~/.bashrc
cp /scratch/ars991/rl_finance_singularity/rl_finance.ext3 ${SINGULARITY_OVERLAY_PATH}
chown ${USER} ${SINGULARITY_OVERLAY_PATH}
```

### Starting up the Singularity Container

In addition to the overlay file from above, you also need to specify a
Singularity container image to run.  The Singularity container image tells
Singularity which operating system (e.g., Ubuntu, RedHat, Arch Linux, etc.) to
simulate, as well as (optionally) additional hardware to virtualize (e.g.,
GPUs).

We are going to use the following Singularity image:
`/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif`

To start up a Singularity container and run a shell within it, run:

```bash
singularity exec --overlay ${SINGULARITY_OVERLAY_PATH}:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash
```

_Note: this tells Singularity to treat the overlay as a read-only filesystem,
i.e., you will NOT be able to write files to the /ext3 directory. To treat the
overlay as a read-write filesystem (which you will need, e.g., if you want to
install new Python packages [see below]), run the following instead:_

```bash
singularity exec --overlay ${SINGULARITY_OVERLAY_PATH}:rw /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash
```

### Working with Python

The first thing you should always do upon starting the Singularity container is
run:

```bash
source /ext3/env.sh
```

This ensures that you will default to using the version of Python that has been
installed to the overlay, as opposed to whatever version you get by default from
the Singularity image.

Once you have done this, make sure that your Python interpreter lives in the
`/ext3` directory; run

```bash
which python
```

and confirm that the output is: `/ext3/miniconda3/bin/python`

In addition, all the key Python packages should be installed to the appropriate
subdirectory of `/ext3`, typically `/ext3/pip_target`. You can verify this as
follows:

```
Singularity> python
Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> pd.__version__
'1.3.0'
>>> pd.__file__
'/ext3/pip_target/pandas/__init__.py'
```

If you want to install additional Python packages with `pip install`, make sure
that you install them in the overlay so that we can all use the same version as
you. To do this, run:

```bash
pip install -t /ext3/pip_target <package_to_install>
```

_Note: this will NOT work if you are running your Singularity container with the
overlay in read-only mode; see section "Starting up the Singularity Container"
above for how to start the container in read-write mode_

## Training with Hydra

Hydra is a Python framework for configuring machine learning projects.

For our purposes, the most useful aspect of Hydra is that it enables you to
easily and automatically run training jobs across a grid of hyperparameters,
just by specifying on the command line which hyperparameter values you would
like to iterate over. This is done using Hydra's _multirun_ functionality.

In order to use Hydra, you must first create a config YAML file. The config file
for this project is `hydra_config.yaml`.

Then, you must write a Python script which reads that config file and uses the
values in it to (e.g.) train a machine learning model.  This Python script must
have exactly one function that is decorated by the `@hydra.main` decorator. Our
Python script is `hydra_training_driver.py`.

To do a multirun, you simply invoke the Python script with the `--multirun`
flag. The remaining arguments to the script should be a key in the YAML file,
followed by a comma-separated list of values. Typically the key will be the name
of a hyperparameter, and the values will be the hyperparameter values for which
you want to train the model. For example:

```bash
python RL-Finance/hydra_training_driver.py --multirun \
    gamma=0.9,0.95,0.99 \
    gae_lambda=0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95
```

The Python script will then iterate over the Cartesian product of the values you
specified and, each time, call the decorated function with the setting of the
hyperparameters given by the current tuple. Any configs not specified on the
command line will default to the values given in the YAML file.

You can learn more about Hydra from [this blog post](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710)

The full Hydra documentation is [here](https://hydra.cc/docs/intro)
