# Smart Crop Disease Identification

> A Flask image-classification prototype with crop-specific Keras models and a browser upload workflow.

**Python · Flask · TensorFlow/Keras · OpenCV · NumPy**

## Application flow

Choose a crop → upload a leaf image → resize and normalize → run its classifier → display the predicted class and model confidence.

The application defines class mappings for **rice, corn, tomato, potato, and wheat**. It reads each model's input dimensions, converts images from BGR to RGB, normalizes pixel values, and uses the highest-scoring output class.

## Features in the source

- Crop-specific pages and prediction routes.
- PNG/JPG/JPEG extension checks.
- Unique upload filenames.
- Dynamic image resizing based on model input shape.
- Prediction label, confidence score, and predefined guidance text.

## Setup and model requirements

```bash
git clone https://github.com/Pandiyaraj2004/smart-crop-Disease-Identification.git
cd smart-crop-Disease-Identification
python -m pip install flask tensorflow opencv-python numpy
```

Before starting, provide every model expected by `app.py`:

| Expected path | Repository status |
| --- | --- |
| `models/corn_model.h5` | Included |
| `models/potato_model.h5` | Included |
| `models/rice_model.h5` | Missing from this path |
| `models/Tomato_model.h5` | Missing from this path |
| `models/wheatDiseaseModel.h5` | Missing from this path |

All five models load at startup, so missing files currently prevent the server from starting. Supply compatible trained artifacts with output ordering matching the source, or explicitly reduce the configured crop set to available models.

```bash
python app.py
```

Then open http://localhost:5000.

## Explore the code

- [app.py](app.py): model configuration, preprocessing, routes, and results.
- [models](models): included saved models.
- [data](data): example leaf images organized by crop and class.
- [corn model.py](corn%20model.py): corn-model training code.

## Evaluation boundary

Displayed confidence is a model output, not a measured real-world accuracy guarantee. The predefined guidance is demonstration content. Field evaluation, training-data provenance, class-order validation, and compatible dependency pinning remain necessary before practical use.

---

Built by **Pandiyaraj A** · [Portfolio](https://pandiyaraj.netlify.app/) · [GitHub](https://github.com/Pandiyaraj2004) · [LinkedIn](https://www.linkedin.com/in/pandiyaraj-a-mca/)
