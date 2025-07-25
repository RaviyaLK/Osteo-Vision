import shap
import numpy as np
import cv2
import base64

def generate_shap_image(model, img_array, image_resized):

    background = img_array[:10] if len(img_array) >= 10 else np.tile(img_array, (10, 1, 1, 1))

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(img_array)


    predicted_index = np.argmax(model.predict(img_array))
    shap_image = shap_values[predicted_index][0]

    shap_image_norm = np.abs(shap_image).sum(axis=-1)
    shap_image_norm /= shap_image_norm.max()

    shap_image_colored = cv2.applyColorMap(np.uint8(255 * shap_image_norm), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    shap_overlay = cv2.addWeighted(img_rgb, 0.6, shap_image_colored, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", shap_overlay)
    return base64.b64encode(buffer).decode()
