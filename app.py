import streamlit as st
# import pickle
# import pandas as pd
# from sklearn import *
from fastai.vision.core import PILImage
from fastai.learner import load_learner


filename = "model.pkl"
model = load_learner('model.pkl')

# Load the image file
img = PILImage.create('cat_1.png')

cats = [ 'cat_1.png', 'cat_2.png', 'cat_3.png' ]
dogs = [ 'dog_1.png', 'dog_2.png', 'dog_3.png' ]

def main():
    st.set_page_config(page_title="Cat/Dog classifier")
    overview = st.container()
    left, right = st.columns(2)

    with overview:
        st.title("Cat/Dog classifier")
        st.write("I conducted five training epochs since further iterations resulted in diminished accuracy, "
                 "and validation loss began to exhibit a slight increase, suggesting the onset of overfitting. "
                 "Utilizing a MacBook equipped with an M1 chip posed challenges as PyTorch relies on GPU instructions "
                 "not compatible with this architecture. Consequently, I resorted to employing the CPU, "
                 "albeit slower, but suitable for the modest dataset size.")


    with left:
        for cat in cats:
            left.image(cat, caption='Image to classify', width=200)
            img = PILImage.create(cat)
            pred_class, pred_idx, probs = model.predict(img)
            left.write(f'Predicted class: {pred_class}')
            left.write(f'Probability: {float(probs[0])}')

    with right:
        for dog in dogs:
            right.image(dog, caption='Image to classify', width=200)
            img = PILImage.create(dog)
            pred_class, pred_idx, probs = model.predict(img)
            right.write(f'Predicted class: {pred_class}')
            right.write(f'Probability: {float(probs[1])}')

    # Use the model to make a prediction
    pred_class, pred_idx, probs = model.predict(img)

    # Print the predicted class and the probabilities
    print(f'Predicted class: {pred_class}')
    print(f'Probabilities: {probs}')


if __name__ == "__main__":
    main()
