# Online Active Learning with Corrective Feedback
Online Active Learning with Corrective Feedback (OAL-CF) is a machine learning training paradigm that allows a model to be trained to perform a specific task alongside a human expert who performs the same task. This repository contains code that implements the OAL-CF pipeline, application to the tasks of Voice-Type Discrimination (VTD) and Spoken Language Verification (SLV), as well as many plotting, scoring, and feature extraction scripts.

For experiments performed in the paper "A Unified Metric for Simultaneous Evaluation of Error Rate and Annotation Cost", jump to the "IMLM Experiments" section.

Note: The Active Learning with Cost Embedding (ALCE) option that was used in the thesis was disabled due to outdated dependency issues.

## Dependencies
Dependencies are contained in the `requirements.txt` file. To equip your virtual environment with these dependencies, start a virtual environment with Python 3.11.9 installed and run this command:

```
pip install -r requirements.txt
```

Additional configuration of Pytorch for your machine may be required.

## Feature Extraction
Python scripts for SLV x-vector, ECAPA, and WavLM feature extraction are located in the `utils` directory (`extract_xvectors.py`, `extract_ecapa.py`, and `extract_wavlm.py`). To extract a feature from an MP3 audio dataset with train/dev/test splits, copy the script to the directory containing the split directories, and run the script there. The output will be placed in a directory called `wavlm`, `xvectors`, or `ecapalang` (depending on the script) with subdirectories called `train`, `test`, and `dev`.

Once the features are extracted, you can generate an order file that specifies the order in which SLV samples are presented. This can be done by copying either `org_cluster.py` or `org_random.py` to the directory containing the split directories and running it there.

## Demonstration
Below are demonstrations of how to start VTD and SLV training/evaluation runs using the OAL-CF paradigm. All commands should be run from the main directory of this repository. 

Parameters listed in `src/params.py` can be controlled via the command line. Note that all parameters related to feature or label directories (in the `Data arguments` section of the parameter file) will need to be adjusted to point to the correct places.

### VTD Example
The full command to train and evaluate a model on the VTD task using the OAL-CF paradigm is shown below. Note that the `--sim_type` flag makes this an OAL-CF run rather than an OAL run.

```
python run.py --run_name test_vtd --sim_type fpstps --max_fb_samples 16
```

The command above will only run a single VTD environment (rm1, mic20 from the SRI Corpus, by default). The following command calls a script that will simultaneously run all standard VTD environments in different screens on two different GPUs. 

```
bash scripts/run_parallel.sh test_vtd "--sim_type fpstps --max_fb_samples 16"
```

### SLV Example—African Continent
Here is the full command to run a single split SLV on the African Continent data. Note that the `--feat_root` parameter now points to a different directory.

```
python run.py --run_name test_aclid --sim_type fpstps --max_fb_samples 16 --env_name test --feat_root /mnt/usb1/AfricanContinentLID/ecapalang/
```

### SLV Example—Caucasus Region
Here is the full command:

```
python run.py --run_name test_aclid --sim_type fpstps --max_fb_samples 16 --env_name test --feat_root /mnt/usb2/CaucasusRegionLID/ecapalang/
```

And here is the script that runs all languages simultaneously:

### Simultaneous Runs
The commands above only runs one language. To run all tests for all languages, use the following bash script:

```
bash scripts/run_lid_parallel.sh test_lid "--sim_type fpstps --max_fb_samples 16"
```

### Budget Planning
To use planned non-uniform AL budgets with the `--budget_path` flag, first create a budget using `utils/plan_budget.py`. Before this can be done, there must be a directory filled with existing runs for the environments to be budgeted. Then, the script can be run as follows:

```
python utils/plan_budget.py [budget_type] path/to/runs/directory [budget_name] [other_parameters]
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

## IMLM Experiments
This repository was used for the illustrative experiments in the paper titled "A Unified Metric for Simultaneous Evaluation of Error Rate and Annotation Cost". To run these OAL experiments, the `scripts/run_icassp_lang.sh` script was used. All parameters for these experiments are found in the script code or the default parameters in `src/params.py`. Here is an example command using the script:

```
bash scripts/run_icassp_lang.sh [lang_id]
```
