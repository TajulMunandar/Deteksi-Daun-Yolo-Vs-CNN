"""
Flask API for YOLOv11 Leaf Detection
======================================
This Flask application provides a REST API for leaf detection inference.

Usage:
    python app.py

Endpoints:
    POST /predict - Upload an image and get detection results
    GET /health - Health check endpoint
    GET /classes - Get list of detectable classes
    GET / - Web frontend

Example curl request:
    curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch

# Configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/detect/leaf_detection/weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))

# Global model variable
model = None

def load_model():
    """Load YOLOv11 model once at startup."""
    global model
    
    print("=" * 60)
    print("Loading YOLOv11 Leaf Detection Model")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        device = 0
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    # Load model
    if os.path.exists(MODEL_PATH):
        print(f"\n📦 Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.to(device)
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️  Model not found at: {MODEL_PATH}")
        print("   Please train the model first.")
        model = None

# Home route - serve frontend
@app.route('/')
def home():
    """Serve the frontend HTML page."""
    return render_template('index.html')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    return jsonify(status)

# Get classes endpoint
@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of detectable classes."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    classes = {str(k): v for k, v in model.names.items()}
    return jsonify({
        "classes": classes,
        "total_classes": len(classes)
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint for leaf detection.
    """
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure the model is trained."}), 500
    
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400
    
    image_file = request.files['image']
    
    # Check file type
    filename = image_file.filename.lower()
    if not filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        return jsonify({"error": "Invalid file type."}), 400
    
    try:
        # Preprocess image
        img = Image.open(image_file)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Run inference
        results = model.predict(
            source=img_array,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        # Process results
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            
            for i, (box, conf, c) in enumerate(zip(boxes, confs, cls)):
                detection = {
                    "detected_class": results.names.get(int(c), f"class_{c}"),
                    "confidence": float(conf),
                    "bounding_box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                detections.append(detection)
        
        # Draw bounding boxes on image
        annotated_img = img_array.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            
            for i, (box, conf, c) in enumerate(zip(boxes, confs, cls)):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[int(c) % len(colors)]
                
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                class_name = results.names.get(int(c), f"class_{c}")
                label = f"{class_name}: {conf:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), font, font_scale,
                           (255, 255, 255), thickness)
        
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            "success": True,
            "predictions": detections,
            "total_detections": len(detections),
            "annotated_image": annotated_img_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Base64 prediction endpoint
@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """Predict endpoint for base64 encoded images."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        image_data = base64.b64decode(data['image'])
        image_file = io.BytesIO(image_data)
        
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        results = model.predict(
            source=img_array,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            
            for i, (box, conf, c) in enumerate(zip(boxes, confs, cls)):
                detection = {
                    "detected_class": results.names.get(int(c), f"class_{c}"),
                    "confidence": float(conf),
                    "bounding_box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                detections.append(detection)
        
        annotated_img = img_array.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            
            for i, (box, conf, c) in enumerate(zip(boxes, confs, cls)):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[int(c) % len(colors)]
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                class_name = results.names.get(int(c), f"class_{c}")
                label = f"{class_name}: {conf:.2f}"
                
                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            "success": True,
            "predictions": detections,
            "total_detections": len(detections),
            "annotated_image": annotated_img_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLOv11 Leaf Detection Flask API")
    print("=" * 60)
    
    # Load model before starting server
    load_model()
    
    print("\n🚀 Starting Flask API Server...")
    print(f"   Server running at: http://localhost:5000")
    print("\n📋 Available Endpoints:")
    print("   GET  /            - Web frontend")
    print("   GET  /health      - Health check")
    print("   GET  /classes     - Get list of classes")
    print("   POST /predict     - Upload image for prediction")
    print("   POST /predict/base64 - Base64 image for prediction")
    print("\n📝 Example curl request:")
    print('   curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict')
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("DEBUG", "False").lower() == "true"
    )
