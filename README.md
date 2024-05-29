# The Interactive Machine Learning Paradigm
The Interactive Machine Learning (IML) paradigm is a machine learning training scheme that allows a model to be trained to perform a specific task alongside a human expert who performs the same task. This repository contains code that implements the IML pipeline, application to the tasks of Voice-Type Discrimination (VTD) and Language Identification (LID), as well as many plotting, scoring, and feature extraction scripts.

## Dependencies
Dependencies are contained in the `requirements.txt` file. To equip your virtual environment with these dependencies, start a virtual environment with Python 3.9.18 installed and run this command:

```
pip install -r requirements.txt
```

Additional configuration of Pytorch for your machine may be required.

## Demonstration
Below are demonstrations of how to start VTD and LID training/evaluation runs using the IML paradigm. All commands should be run from the main directory of this repository. 

Parameters listed in `src/params.py` can be controlled via the command line. Note that all parameters related to feature or label directories (in the `Data arguments` section of the parameter file) will need to be adjusted to point to the correct places.

### VTD Example
The full command to train and evaluate a model on the VTD task using the IML paradigm is shown below. Note that the `--sim_type` flag makes this an IML run rather than an OAL run.

```
python run.py --run_name test_vtd --sim_type fpstps --max_fb_samples 16
```

The command above will only run a single VTD environment (rm1, mic20 from the SRI Corpus, by default). The following command calls a script that will simultaneously run all standard VTD environments in different screens on two different GPUs. 

```
bash scripts/run_parallel.sh test_vtd "--sim_type fpstps --max_fb_samples 16"
```

### LID Example—African Continent
Here is the full command to run a single split LID on the African Continent data. Note that the `--feat_root` parameter now points to a different directory.

```
python run.py --run_name test_aclid --sim_type fpstps --max_fb_samples 16 --env_name dev --feat_root /mnt/usb1/AfricanContinentLID/wavlm/,/mnt/usb1/AfricanContinentLID/xvectors/ --context 0
```

The command above only runs the development split. To run both development and test splits simultaneously, use this script:

```
bash scripts/run_aclid_parallel.sh test_aclid "--sim_type fpstps --max_fb_samples 16"
```

### LID Example—Caucasus Region
Again, here is the full command:

```
python run.py --run_name test_aclid --sim_type fpstps --max_fb_samples 16 --env_name dev --feat_root /mnt/usb2/CaucasusRegionLID/wavlm/,/mnt/usb2/CaucasusRegionLID/xvectors/ --context 0
```

And here is the script that runs both splits simultaneously:

```
bash scripts/run_aclid_parallel.sh test_aclid "--sim_type fpstps --max_fb_samples 16"
```
