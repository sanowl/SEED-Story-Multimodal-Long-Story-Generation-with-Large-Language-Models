import streamlit as st
import jax.numpy as jnp
from models.seed_story import SEEDStory
from config import SEEDStoryConfig
from image_utils import load_and_preprocess_image
import tempfile
from PIL import Image
import io

@st.cache(allow_output_mutation=True)
def load_model(config_path, checkpoint_path):
    config = SEEDStoryConfig.from_json(config_path)
    model = SEEDStory(config)
    model = model.load_checkpoint(checkpoint_path)
    return model

def main():
    st.title("SEEDStory: Image-to-Story Generation")

    # Load model
    model = load_model("config.json", "model_checkpoint.ckpt")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        preprocessed_image = load_and_preprocess_image(tmp_file_path)
        preprocessed_image = jnp.array(preprocessed_image)[None, ...]  # Add batch dimension

        # Get user prompt
        prompt = st.text_input("Enter a story prompt:", "Once upon a time,")

        if st.button("Generate Story"):
            with st.spinner("Generating story..."):
                generated_story = model.apply({'params': model.params}, (preprocessed_image, prompt))
                st.write("Generated Story:")
                st.write(generated_story)

        # Advanced options
        with st.expander("Advanced Options"):
            temperature = st.slider("Temperature", 0.1, 1.0, 0.8, 0.1)
            max_length = st.slider("Max Length", 50, 500, 100, 10)

            if st.button("Generate Story with Custom Settings"):
                with st.spinner("Generating story..."):
                    generated_story = model.apply(
                        {'params': model.params},
                        (preprocessed_image, prompt),
                        method=model.generate_story,
                        generate_kwargs={"temperature": temperature, "max_length": max_length}
                    )
                    st.write("Generated Story (Custom Settings):")
                    st.write(generated_story)

if __name__ == "__main__":
    main()