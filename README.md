# resume-screener
📄 🔎 Screen resumes using a job description and LLMs

1. You will need an OpenAI API key
2. Update `job_spec.yml` to match your job requirements
3. Update the `UpdatedResumeDataSet.csv` to match the resumes you would like to screen.

```bash
pip install -r resume-screener/requirements.txt
export OPENAI_API_KEY=*******your***key***
python resume-screener/screener.py
```
