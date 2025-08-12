# LLM model scaling pipeline

## For (s)LLM model scaling

### Dataset
HuggingFace Korean dataset(preprocessed as instruction, input, and response)

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/chatbot-demo.git
cd chatbot-demo

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting
```shell
OPENAI_API_KEY={OPENAI_API_KEY}
SERPAPI_API_KEY={SERPAPI_API_KEY}
```

### Run

* Run demo
```shell
streamlit run app.py
```