# FastAPI Knee Osteoporosis Detection

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `python main.py`
3. Endpoints:
   - `/upload/` → Predicts osteoporosis & generates Grad-CAM
   - `/bone_density/` → Estimates bone density
   - `/cortical_thickness/` → Measures cortical thickness
