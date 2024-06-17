# The Interactive Machine Learning Paradigm
The Interactive Machine Learning (IML) paradigm is a machine learning training scheme that allows a model to be trained to perform a specific task alongside a human expert who performs the same task. This repository contains code that implements the IML pipeline, application to the tasks of Voice-Type Discrimination (VTD) and Language Identification (LID), as well as many plotting, scoring, and feature extraction scripts.

## Dependencies
Dependencies are contained in the `requirements.txt` file. To equip your virtual environment with these dependencies, start a virtual environment with Python 3.9.18 installed and run this command:

```
pip install -r requirements.txt
```

Additional configuration of Pytorch for your machine may be required.

## Feature Extraction
Python scripts for LID x-vector, ECAPA, and WavLM feature extraction are located in the `utils` directory (`extract_xvectors.py`, `extract_ecapa.py`, and `extract_wavlm.py`). To extract a feature from an MP3 audio dataset with train/dev/test splits, copy the script to the directory containing the split directories, and run the script there. The output will be placed in a directory called `wavlm`, `xvectors`, or `ecapalang` (depending on the script) with subdirectories called `train`, `test`, and `dev`.

Once the features are extracted, you can generate an order file that specifies the order in which LID samples are presented. This can be done by copying either `org_cluster.py` or `org_random.py` to the directory containing the split directories and running it there.

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
python run.py --run_name test_aclid --sim_type fpstps --max_fb_samples 16 --env_name dev --feat_root /mnt/usb1/AfricanContinentLID/ecapalang/
```

The command above only runs the development split. To run both development and test splits simultaneously, use this script:

```
bash scripts/run_aclid_parallel.sh test_aclid "--sim_type fpstps --max_fb_samples 16"
```

### LID Example—Caucasus Region
Again, here is the full command:

```
python run.py --run_name test_aclid --sim_type fpstps --max_fb_samples 16 --env_name dev --feat_root /mnt/usb2/CaucasusRegionLID/ecapalang/
```

And here is the script that runs both splits simultaneously:

```
bash scripts/run_aclid_parallel.sh test_aclid "--sim_type fpstps --max_fb_samples 16"
```

## Scoring
The scoring scripts are all contained in the `scoring` directory. The two most commonly used scripts here will be `print_basic_results.py` and `combo_scores.py`. To see the scores of a single run, use this command:

```
python scoring/print_basic_results.py output/run_dir/run_name/
```

To see the aggregated scores for multiple runs with the same parameter settings (i.e., runs initiated with one of the bash scripts), use this command:

```
python scoring/print_basic_results.py output/run_dir/*
```
