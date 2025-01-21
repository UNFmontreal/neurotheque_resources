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
│   │   └── analysis/
│   ├── preprocessing/
│   │   ├──global_cleaning.py 
│   │   └──task_preprocessing.py
│   └── analysis/                # Pre-made analysis pipelines
│       ├── gonogo_analysis.py
│       ├── finger_tap_analysis.py
│       ├── mental_imagery_analysis.py
│       └── landoit_c_analysis.py
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
├── checkpoints/ 
├── tests/
│   ├── test_pipeline.py
│   ├── test_steps.py
│   └── test_strategies.py
├── README.md
├── requirements.txt
└── setup.py (or pyproject.toml)



neuropipe/
├── src/
│   ├── pipeline.py              # Pipeline class (executes steps in order)
│   ├── steps/                   # Individual, reusable step definitions
│   │   ├── base.py              # BaseStep (abstract class)
│   │   ├── load.py              # LoadData step
│   │   ├── filter.py            # FilterStep
│   │   ├── ica.py               # ICAStep
│   │   ├── autoreject.py        # AutoRejectStep
│   │   ├── triggers.py          # TriggerParsingStep
│   │   ├── epoching.py          # EpochingStep
│   │   ├── plot.py (optional)   # Plotting steps
│   │   ├── save.py              # SaveData step
│   │   └── checkpoint.py (optional) # CheckpointStep
│   ├── preprocessing/           # Pre-made pipelines or “strategies” for data cleaning
│   │   ├── global_and_task_preprocessing.py 
│   │   └── ...
│   ├── analysis/                # Pre-made pipelines or “strategies” for task-level analysis
│   │   ├── gonogo_analysis.py
│   │   ├── finger_tap_analysis.py
│   │   ├── mental_imagery_analysis.py
│   │   └── landoit_c_analysis.py
│   ├── strategies/              # Alternative location for “strategy” scripts
│   │   ├── basic_cleaning.py
│   │   ├── gonogo_strategy.py
│   │   └── ...
│   └── ...
├── configs/                     # YAML config files for pipelines
│   ├── basic_cleaning.yml
│   ├── gonogo_example.yml
│   └── ...
├── checkpoints/                 # (Optional) folder for partial-run checkpoint files
├── tests/                       # Unit & integration tests
│   └── ...
├── README.md
├── requirements.txt
└── setup.py (or pyproject.toml)
