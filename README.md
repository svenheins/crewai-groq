# crewai-groq
Running a mixture of agents crew with crewAI and groq. This project is a migration of the already existing example https://github.com/groq/groq-api-cookbook/tree/main/tutorials/crewai-mixture-of-agents but there was a lack of documentation regarding the installation. This is addressed in this repository.

## installation
* python==3.11
* the package statsapi ahs been moved to MLB-StatsAPI

## package management with conda
```
conda create -n crewai-groq
conda activate crewai-groq
pip install -r requirements
```

## run
```
cp .env.example .env
```
Get your free groq token: https://console.groq.com/ and insert it into the .env file. ***Important***: Please ensure that you don't share your dev token (i.e. forking this repo and put it into the .env.example)

Now you can select your python interpreter within your IDE and execute the code
