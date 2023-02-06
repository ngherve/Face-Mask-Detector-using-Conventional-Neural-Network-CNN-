import io

import streamlit as st
from appInference import *
from evaluation import evaluate


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model():
    train_loader, test_loader, my_categories = pre_processing()
    model = torch.load(top_path + '/trained_model_3.pth')
    evaluate(model, test_loader, my_categories)
    return model


def load_labels():
    labels_path = 'imagemasks_classes.txt'
    labels_file = os.path.basename(labels_path)

    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    outputs, max_elements, max_idxs = Test(model, image)
    st.write(categories[max_idxs])


def main():
    st.title('Pretrained model demo')
    categories = load_labels()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(top_path + '/trained_model_3.pth', categories, image)


if __name__ == '__main__':
    main()
