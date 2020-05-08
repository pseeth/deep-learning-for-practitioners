# Suggested code structure

Here's a suggested structure for your repository that is reproducing your chosen paper.

## Overview

There are four main components to structuring a repository that contains experiments in
machine learning or data science:

- Building blocks: these are the building blocks that you use to train, build, deploy, test
  any of your models and experiments. Building blocks are things like a trainer class, builders
  for models, data handling utilities (e.g. something subclassing `torch.utils.data.Dataset`), 
  and so on. Nothing in this component should be a script. Everything should be either a function,
  a class, or a variable.
- Experiment code and scripts: these contain actual experiments or other scripts (e.g. setting up
  or downloading data for experiments). Experiments use things from building blocks to run. If you
  design your building blocks correctly, then experiment code should be pretty compact, extensible,
  and easy to read and understand.
- Tests: this should contain actual tests that look at your building blocks and make sure they do
  what you think they are doing. You should use something like `pytest` and `pytest-cov` here, 
  (https://pytest-cov.readthedocs.io/en/latest/reporting.html) and design your tests to really put
  your code through the ringer. Make sure it works. If you find a bug through some other means, make
  sure the test is updated to catch the bug, THEN fix the code. You should get into a cycle of test
  driven development.
- Notebooks: Some of you may be familiar with Jupyter notebooks. You can use notebooks as a quick way 
  to interact with building blocks, visualize the results of experiments, etc. I use them as a way to 
  rapidly prototype a building block or test before putting it officially into the code. Generally,
  I keep a notebook server running at all times to help with this.
  
## Structure

The four main components should all be present in your repository. Here's a suggested outline for how
to get them all in there:

```
├── LICENSE             # Maybe MIT, BSD3, see https://choosealicense.com/
├── README.md           # A solid README explaining the project and how to use the code
├── conda.yml           # Could be a conda.yml or a requirements.txt file that gets you dependencies.
├── experiments
│   ├── prepare_data.py # Might be a script that downloads/prepares data for processing.
│   ├── exp1.py         # Experiment 1 might reproduce table 1. Uses stuff from src.
│   └── exp2.py         # Might have many experiments. 
├── notebooks
│   └── demo.ipynb      # Notebook demoing the model? Whatever it is.
├── src                 # Building blocks. These are not set in stone, just examples.
│   ├── dataset.py      # Data handling
│   ├── handlers.py     # Any handlers during training
│   ├── model.py        # Model building
│   ├── evaluate.py     # Evaluation building blocks
│   └── train.py        # Training building blocks
└── tests               # Test your building blocks.
    ├── test_dataset.py     
    ├── test_handlers.py 
    ├── test_model.py
    ├── test_evaluate.py
    └── test_train.py
```

This is just an example of how to structure it. Yours might look a bit different. 
But it's important that all the building blocks are there.
