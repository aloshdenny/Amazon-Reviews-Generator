This repo doesn't have the model, per se. You have to train it locally as the model is quite large to upload on GitHub.
All the weights and biases have been pre-tuned in the **create_model.ipynb** notebook.
The code is set to run on **GPU** by default, so if any issues arise, please refer **PyTorch** help forum.
Run the code in above said notebook.
It took ~7ish hours to train the model on my **RTX 3050**, but you can optimise it by fine-tuning overfit and underfit params, epochs and batch_size.
On a decent 4-core CPU, it should take ~12 hours and on an 8-core CPU, ~10 hours.
