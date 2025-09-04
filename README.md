# LLM model scaling pipeline

## For (s)LLM model scaling

### Dataset

Cosmecca Recipe dataset

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/recipe-ai-test.git
cd recipe-ai-test

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting

```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}

API_KEY={API_KEY}
REMOTE_API_BASE_RECOMMEND={REMOTE_API_BASE_RECOMMEND}
REMOTE_API_BASE_REPORT={REMOTE_API_BASE_REPORT}
```

### Run

* Run demo

```shell
streamlit run app.py
```
