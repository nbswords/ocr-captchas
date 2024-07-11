# ocr-captchas

A pratice of OCR model for recognition Captchas by Tensorflow.Keras

## Usage

```shell
# Train and save model
python ocr_captcha.py
```

```shell
# Inference with single image named `test.png`
python predict_example.py
```

## Data
Dataset is from Kaggle's CAPTCHA Images.</br>

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