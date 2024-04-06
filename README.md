# Onfido EyeGoogler

This project is a simple web service that allows users to upload a photo of peoples faces and get back the image with
fun googly eyes!

The project uses a pretrained model called BlazeFace available from Google in their MediaPipe
package (https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf).
It's optimised for on device applications so is very small and quite fast, which works well as a first pass for this
application.
Although depending on requirements it could be worth using a larger model if it's going to run server side where more
compute is available rather than client side.

## Setup

The project uses Python 3.10.5 and Poetry (https://github.com/python-poetry/poetry) for environment management, these
should be set up on your sytem first.

1. Install the Poetry environment

```shell
poetry install 
```

2. Launch the server

```shell
poetry run  uvicorn eyegoogler.api.main:app
```

3. Use the test image to test

```shell
curl -v -F "file=@image.jpg" http://127.0.0.1:8000/eyegoogler/ --output output.png
```

## API Spec

The API exposes a single endpoint `/eyegoogler` that expects to receive image data as a file upload.
The image data is loaded by OpenCV, so
all [formats supported](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8) by
OpenCV could be accepted (only tested JPEG and PNG).

The API returns PNG image data directly to the client.

## Developing

### Ruff

[Ruff](https://docs.astral.sh/ruff/) is used as a code formatter and can be triggered using

```shell
poetry run ruff format
```

### Pytest

Pytest is used as a testing framework, to run the tests from the root of the repo run

```shell
poetry run pytest
```

## Further work for production

* **Model evaluation** - Before releasing this we would need to test more rigorously to understand the performance of
  the face detection model and understand how it behaves in more challenging cases, eg side views, obscured views or
  pictures with no faces.
  There are also fairness/bias considerations eg does it work equal well with different ages, genders and ethnicities.

* **Model development** - This prototype uses a pretrained model that is optimised to be lightweight,
  if its performance is not sufficient for the application there is potential to use a larger architecture or fine tune
  a model for the task.

* **Logging, monitoring and alerting** - Are essential for a production service - eg monitoring the performance of the
  different components, alerting on degradation like exceptions, latency and 'no faces detected' event. Logging samples
  of inputs and outputs for manual review or building future datasets.

* **Deployment** - Deployment to kubernetes would work well here as the service is stateless and horizontally scalable.
  Possibility to add autoscaling based on CPU/queue metrics. Could also look at a dedicated model serving solution like
  Triton.

* **Load/perf testing** - Unscientific test gave me ~80ms latency on my laptop, which likely could be improved on.
  Configuration of uvicorn would optimise concurrency and likely would switching to async handlers.  
 