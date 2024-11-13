# resume-screener
📄 🔎 Screen resumes using a job description and LLMs

## How It works
1. It embeds your resumes and caches the embeddings locally
2. Then it takes a job-spec and finds the top resumes in terms of keywords and symantic similarity
3. Then it screens the top N resumes against the must-have section of the job-spec using an LLM.

🕝️ It takes a minute the first time you run it because it puts embeddings in a local cache, but after that it will run quickly if you need to adjust the prompt or job description.

## Getting Started
1. You will need an OpenAI API key
2. Update `job_spec.yml` to match your job requirements
3. Update the `UpdatedResumeDataSet.csv` to match the resumes you would like to screen.

```bash
pip install -r resume-screener/requirements.txt
export OPENAI_API_KEY=*******your***key***
python resume-screener/screener.py
```

This script uses Langchain, so if you want to switch from an OpenAI model to Anthropic, Groq, Cohere, or many others this is a simple change.
