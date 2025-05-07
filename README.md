# medical-chatbot
# End-to-end-Medical-Chatbot-hugging-face-embeddings-mistral

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/devs6186/medical-chatbot.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.10 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MISTRAL_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
FLASK_SECRET_KEY= "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

model_name='sentence-transformers/all-MiniLM-L6-v2'
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Mistral
- Pinecone

