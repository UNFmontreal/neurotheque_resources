# FAQ

Frequently Asked Questions about Neuroflow.

**Q: How do I add a new step to the pipeline?**

A: To add a new step, create a new Python file in the `scr/steps` directory. The file should contain a class that inherits from `BaseStep` and implements a `run` method. The step will then be automatically available to be included in your `config.yml`.

**Q: Can I run the pipeline on data that is not in BIDS format?**

A: It's highly recommended to convert your data to BIDS format first using the provided `dsi_to_bids.py` script or by other means. While the pipeline might work with other data structures, it's designed to work best with BIDS.
