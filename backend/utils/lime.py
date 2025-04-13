import numpy as np
import cv2
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries

def generate_lime_image(model, img_array, image_resized, model_type="multiclass"):
    def predict_fn(images):
        return model.predict(images)

    # Initialize the explainer
    explainer = lime_image.LimeImageExplainer()


    # Run explanation (reduced num_samples for speed)
    explanation = explainer.explain_instance(
        np.array(image_resized),
        predict_fn,
        top_labels=1,     # Only focus on top label for speed
        hide_color=0,
        num_samples=50    # Reduced for faster computation
    )

    label = explanation.top_labels[0]

    # Get visualization image and mask
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Overlay LIME boundary
    lime_output = mark_boundaries(temp / 255.0, mask)

    # Convert to uint8 for encoding
    lime_uint8 = np.uint8(lime_output * 255)
    lime_bgr = cv2.cvtColor(lime_uint8, cv2.COLOR_RGB2BGR)

    # Encode to base64
    _, buffer = cv2.imencode(".jpg", lime_bgr)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return encoded_image
