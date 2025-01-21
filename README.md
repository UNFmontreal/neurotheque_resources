# neuropipe
eeg_pipeline/
├── src/
│   ├── pipeline.py         # The main Pipeline class (reads pipeline steps from config)
│   ├── steps/              # Individual, reusable step definitions
│   │   ├── base.py
│   │   ├── load.py
│   │   ├── filter.py
│   │   ├── ica.py
│   │   ├── autoreject.py
│   │   ├── triggers.py
│   │   ├── epoching.py
│   │   ├── plot.py
│   │   ├── save.py
│   │   └── ...
│   ├── strategies/         # A folder of "pre-made strategy" scripts/classes
│   │   ├── basic_cleaning.py
│   │   ├── gonogo_strategy.py
│   │   ├── finger_tapping_strategy.py
│   │   ├── mental_imagery_strategy.py
│   │   └── landoit_c_strategy.py
├── configs/
│   ├── basic_cleaning.yml
│   ├── gonogo_example.yml
│   ├── finger_tapping_example.yml
│   ├── mental_imagery_example.yml
│   └── landoit_c_example.yml
├── tests/
│   ├── test_pipeline.py
│   ├── test_steps.py
│   └── test_strategies.py
├── README.md
├── requirements.txt
└── setup.py (or pyproject.toml)
