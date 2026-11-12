## Changes made for the Docker Lab 1 assignment

### `src/main.py`
- Switched from the Iris dataset to scikit-learn's built-in Digits dataset: 1,797 small (8x8) images of handwritten digits 0 to 9, each with 64 pixel values.
- Replaced the Random Forest with a Support Vector Machine (SVC, RBF kernel). It scored 0.994 test accuracy on this dataset, compared with 0.961 for the original Random Forest.
- The script now prints the test accuracy. The original code split off a test set but never used it.
- The model is saved as `model/digits_model.pkl`. If you run the container with `-v "$(pwd)/model:/app/model"`, the file ends up in a `model/` folder on your computer. Previously the model stayed inside the container and was lost when the container was removed.

### `src/requirements.txt`
- Pinned scikit-learn (1.5.1) and joblib (1.4.2) so that rebuilding the image later installs the same versions.

### `dockerfile`
- Switched the base image to `python:3.10-slim`. The image shrank from 2.06 GB to 555 MB, and the `docker save` file from 519 MB to 118 MB.
- `requirements.txt` is now copied and installed before the code. Docker caches that step, so a code change no longer triggers a package reinstall, and the rebuild took 0.5 seconds instead of 15.7.
- Added `--no-cache-dir` to `pip install` to keep pip's download cache out of the image.

### `.gitignore` (new)
- Keeps `my_image.tar` (created by the `docker save` command below) and the `model/` folder out of Git. Even with the smaller image, the tar file is 118 MB, over GitHub's 100 MB file limit.

To run this version:

```bash
docker build -t lab1:v1 .
docker run -v "$(pwd)/model:/app/model" lab1:v1
```
---

docker build -t lab1:v1 .

docker save lab1:v1 > my_image.tar

docker run lab1:v1
