python -m streamlit run app/streamlit_app.py


fake-news-detection/
│
├── app/                      # Streamlit application
│   └── streamlit_app.py      # Streamlit UI script
│
├── src/                      # Source code for the project
│   ├── __init__.py           # Makes src a Python module
│   ├── agents/               # Agents for processing news
│   │   ├── __init__.py       # Makes agents a Python module
│   │   ├── class_agent.py    # Classification agent
│   │   ├── decision_agent.py # Decision-making agent
│   │   ├── filter_agent.py   # Filtering agent
│   │   └── headline_agent.py # Headline analysis agent
│   │
│   ├── data_processing/      # Data processing utilities
│   │   ├── __init__.py       # Makes data_processing a Python module
│   │   └── text_preprocessor.py # Text preprocessing utilities
│   │
│   ├── ml_models/            # Machine learning models and pipelines
│   │   ├── __init__.py       # Makes ml_models a Python module
│   │   └── ml_pipeline.py    # ML model and pipeline
│   │
│   └── utils/                # Utility functions and classes
│       ├── __init__.py       # Makes utils a Python module
|       |── info_extraction.py# Information extraction utilities
│       └── token_controler.py# Limit the size of user's inputs to avoid token overflow
|
│
├── requirements.txt          # Project dependencies
└── README.md                 # Project overview and instructions

