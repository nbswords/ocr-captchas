# ocr-captchas

A pratice of OCR model for recognition Captchas by Tensorflow.Keras

## Usage

- Prepare your data and name the folder `captcha_images`

- Train and save the model

```shell
python ocr_captcha.py
```

- Inference with single image named `test.png`

```shell
python predict_example.py
```

## Data
Example Dataset is from Kaggle's CAPTCHA Images.</br>

[Link](https://www.kaggle.com/fournierp/captcha-version-2-images)

## Result

![result](./result.jpg)
</br>

## Learning curve

![learning_curve](./learning_curve.jpg)
</br>

## model
CRNN + CTC loss

![model_structure](./model_structure.jpg)
</br>