"""
Flask API for YOLOv11 and CNN Leaf Detection/Classification
============================================================
This Flask application provides a REST API for leaf detection (YOLO) 
and leaf classification (CNN) for thesis comparison.

Usage:
    python app.py
    
Endpoints:
    POST /predict/yolo - YOLOv11 detection
    POST /predict/cnn - CNN classification
    POST /predict - Unified endpoint (auto-detect)
    GET /health - Health check endpoint
    GET /classes - Get list of detectable/classifiable classes
    GET /compare - Compare both models

Note: YOLOv11 provides object detection (bounding boxes + classification)
      CNN provides image-level classification only
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
from torchvision import transforms
from werkzeug.datastructures import FileStorage

# Import CNN model
from cnn_model import create_model, load_model


# Configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# YOLO Model configuration
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "runs/detect/leaf_detection/weights/best.pt")
YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))

# CNN Model configuration
CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "runs/cnn/best_model.pth")
CNN_MODEL_TYPE = os.environ.get("CNN_MODEL_TYPE", "custom")
CNN_IMG_SIZE = int(os.environ.get("CNN_IMG_SIZE", "224"))

# Global model variables
yolo_model = None
cnn_model = None
cnn_classes = ['daun jeruk', 'daun kari', 'daun kunyit', 'daun pandan', 'daun salam']
device = None


def load_yolo_model():
    """Load YOLOv11 model once at startup."""
    global yolo_model
    
    print("=" * 60)
    print("Loading YOLOv11 Leaf Detection Model")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using GPU: {device_name}")
    else:
        device_name = "CPU"
        print("💻 Using CPU")
    
    # Load model
    if os.path.exists(YOLO_MODEL_PATH):
        print(f"\n📦 Loading YOLO model from: {YOLO_MODEL_PATH}")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO model loaded successfully!")
    else:
        print(f"⚠️  YOLO model not found at: {YOLO_MODEL_PATH}")
        print("   Please train the YOLO model first.")
        yolo_model = None
    
    return yolo_model


def load_cnn_model():
    """Load CNN model once at startup."""
    global cnn_model, device
    
    print("=" * 60)
    print("Loading CNN Leaf Classification Model")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU")
    
    # Load model
    if os.path.exists(CNN_MODEL_PATH):
        print(f"\n📦 Loading CNN model from: {CNN_MODEL_PATH}")
        try:
            cnn_model = load_model(CNN_MODEL_PATH, model_type=CNN_MODEL_TYPE,
                                   num_classes=len(cnn_classes), device=device)
            cnn_model.eval()
            print("✅ CNN model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load CNN model: {e}")
            cnn_model = None
    else:
        print(f"⚠️  CNN model not found at: {CNN_MODEL_PATH}")
        print("   Please train the CNN model first.")
        cnn_model = None
    
    return cnn_model


# Home route - serve frontend
@app.route('/')
def home():
    """Serve the frontend HTML page."""
    return render_template('index.html')


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if (yolo_model is not None or cnn_model is not None) else "model_not_loaded",
        "yolo_loaded": yolo_model is not None,
        "cnn_loaded": cnn_model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {
            "yolo": {"status": "loaded" if yolo_model else "not_loaded"},
            "cnn": {"status": "loaded" if cnn_model else "not_loaded"}
        }
    })


# Debug/test endpoint without model processing
@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Simple test endpoint to verify server is responding."""
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "method": request.method,
        "yolo_loaded": yolo_model is not None,
        "cnn_loaded": cnn_model is not None
    })


# Get classes endpoint
@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of detectable/classifiable classes."""
    return jsonify({
        "classes": {str(i): name for i, name in enumerate(cnn_classes)},
        "total_classes": len(cnn_classes),
        "model_info": {
            "yolo": "Object detection (bounding boxes + classification)",
            "cnn": "Image-level classification only"
        }
    })


# Helper function for YOLO prediction
def _predict_yolo_internal(image_file, include_image=True):
    """Internal YOLO prediction function."""
    global yolo_model
    
    if yolo_model is None:
        return None, "YOLO model not loaded"
    
    try:
        # Reset file position
        image_file.seek(0)
        
        # Preprocess image
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        # Run YOLO inference
        results = yolo_model.predict(
            source=img_array,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
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
                    "detected_class": yolo_model.names.get(int(c), f"class_{c}"),
                    "confidence": float(conf),
                    "bounding_box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1])
                    }
                }
                detections.append(detection)
        
        # Create visualization only if requested
        annotated_img_base64 = None
        if include_image:
            annotated_img = img_array.copy()
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy()
                
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                
                for box, conf, c in zip(boxes, confs, cls):
                    x1, y1, x2, y2 = box.astype(int)
                    color = colors[int(c) % len(colors)]
                    
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    class_name = yolo_model.names.get(int(c), f"class_{c}")
                    label = f"{class_name}: {conf:.2f}"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    cv2.rectangle(annotated_img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated_img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
            
            _, buffer = cv2.imencode('.jpg', annotated_img)
            annotated_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            "success": True,
            "model": "yolov11",
            "task": "object_detection",
            "predictions": detections,
            "total_detections": len(detections)
        }
        
        if annotated_img_base64:
            result["annotated_image"] = annotated_img_base64
        
        return result, None
        
    except Exception as e:
        import traceback
        print(f"❌ YOLO Prediction error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, str(e)


# Helper function for CNN prediction
def _predict_cnn_internal(image_file, include_image=True):
    """Internal CNN prediction function."""
    global cnn_model, device
    
    if cnn_model is None:
        return None, "CNN model not loaded"
    
    try:
        # Reset file position
        image_file.seek(0)
        
        # Preprocess image
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Transform for CNN
        transform = transforms.Compose([
            transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run CNN inference
        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        # Get prediction details
        pred_class = cnn_classes[predicted.item()]
        pred_confidence = confidence.item()
        
        probabilities = {
            cnn_classes[i]: float(probs[0][i])
            for i in range(len(cnn_classes))
        }
        
        # Get top 3 predictions
        top_predictions = sorted(
            [{"class": k, "probability": v} for k, v in probabilities.items()],
            key=lambda x: x['probability'],
            reverse=True
        )[:3]
        
        # Create visualization only if requested
        annotated_img_base64 = None
        if include_image:
            # Create visualization
            img_array = np.array(img)
            
            # Draw classification result
            overlay = img_array.copy()
            cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_array, 0.4, 0, img_array)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_array, f"CNN Classification", (20, 35), font, 0.7, (0, 255, 0), 2)
            cv2.putText(img_array, f"Class: {pred_class}", (20, 65), font, 0.6, (255, 255, 255), 1)
            cv2.putText(img_array, f"Confidence: {pred_confidence*100:.1f}%", (20, 90), font, 0.6, (255, 255, 255), 1)
            
            # Draw probability bar
            bar_x, bar_y = 20, 105
            bar_width = 200
            bar_height = 15
            
            cv2.rectangle(img_array, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(img_array, (bar_x, bar_y), 
                         (bar_x + int(bar_width * pred_confidence), bar_y + bar_height), 
                         (0, 255, 0), -1)
            
            _, buffer = cv2.imencode('.jpg', img_array)
            annotated_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            "success": True,
            "model": "cnn",
            "task": "image_classification",
            "predicted_class": pred_class,
            "confidence": pred_confidence,
            "probabilities": probabilities,
            "top_predictions": top_predictions
        }
        
        if annotated_img_base64:
            result["annotated_image"] = annotated_img_base64
        
        return result, None
        
    except Exception as e:
        import traceback
        print(f"❌ CNN Prediction error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, str(e)


# YOLO Prediction endpoint
@app.route('/predict/yolo', methods=['POST'])
def predict_yolo():
    """
    YOLOv11 Prediction endpoint for leaf detection.
    
    Parameters:
        - image: The image file to detect
        - include_image: 'true' (default) or 'false' to skip annotated image
    
    Returns bounding boxes with classifications.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    
    # Check file type
    filename = image_file.filename.lower()
    if not filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Check if we should include the annotated image
    include_image = request.form.get('include_image', 'false').lower() != 'false'
    
    result, error = _predict_yolo_internal(image_file, include_image=include_image)
    
    if error:
        return jsonify({"error": f"YOLO prediction failed: {error}"}), 500
    
    return jsonify(result)


# CNN Prediction endpoint
@app.route('/predict/cnn', methods=['POST'])
def predict_cnn():
    """
    CNN Prediction endpoint for leaf classification.
    
    Parameters:
        - image: The image file to classify
        - include_image: 'true' (default) or 'false' to skip annotated image
    
    Returns single image-level classification.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    
    # Check file type
    filename = image_file.filename.lower()
    if not filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Check if we should include the annotated image
    include_image = request.form.get('include_image', 'true').lower() != 'false'
    
    result, error = _predict_cnn_internal(image_file, include_image=include_image)
    
    if error:
        return jsonify({"error": f"CNN prediction failed: {error}"}), 500
    
    return jsonify(result)


# Unified prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Unified prediction endpoint - uses both YOLO and CNN.
    """
    model_type = request.form.get('model', 'both')
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    image_data = image_file.read()
    image_filename = image_file.filename
    
    # Check if we should include the annotated image
    include_image = request.form.get('include_image', 'false').lower() != 'false'
    
    if model_type == 'yolo':
        # Reset file position for YOLO
        image_file = FileStorage(io.BytesIO(image_data), filename=image_filename)
        result, error = _predict_yolo_internal(image_file, include_image=include_image)
        if error:
            return jsonify({"error": f"YOLO prediction failed: {error}"}), 500
        return jsonify(result)
    
    elif model_type == 'cnn':
        # Reset file position for CNN
        image_file = FileStorage(io.BytesIO(image_data), filename=image_filename)
        result, error = _predict_cnn_internal(image_file, include_image=include_image)
        if error:
            return jsonify({"error": f"CNN prediction failed: {error}"}), 500
        return jsonify(result)
    
    elif model_type == 'both':
        # Run both models
        yolo_result, yolo_error = None, None
        cnn_result, cnn_error = None, None
        
        # YOLO
        yolo_file = FileStorage(io.BytesIO(image_data), filename=image_filename)
        yolo_result, yolo_error = _predict_yolo_internal(yolo_file, include_image=include_image)
        
        # CNN
        cnn_file = FileStorage(io.BytesIO(image_data), filename=image_filename)
        cnn_result, cnn_error = _predict_cnn_internal(cnn_file, include_image=include_image)
        
        return jsonify({
            "success": True,
            "model_type": "both",
            "yolo_detection": yolo_result,
            "cnn_classification": cnn_result,
            "comparison_note": "YOLO provides bounding boxes for multiple objects. CNN provides single image classification."
        })
    
    else:
        return jsonify({"error": f"Unknown model type: {model_type}"}), 400


# Compare models endpoint
@app.route('/compare', methods=['GET'])
def compare_models():
    """
    Compare YOLO and CNN models.
    
    Returns model architecture details and capabilities comparison.
    """
    return jsonify({
        "models": {
            "yolov11": {
                "task": "Object Detection",
                "description": "YOLOv11 provides both object detection (bounding boxes) and classification",
                "capabilities": [
                    "Detect multiple objects in single image",
                    "Provide bounding box coordinates",
                    "Classify each detected object",
                    "Output confidence scores per detection"
                ],
                "output_format": {
                    "detections": "List of objects with bounding_box, class, confidence",
                    "annotated_image": "Image with drawn bounding boxes"
                },
                "model_path": YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else None,
                "model_loaded": yolo_model is not None
            },
            "cnn": {
                "task": "Image Classification",
                "description": "CNN provides single image-level classification",
                "capabilities": [
                    "Classify entire image",
                    "Output probabilities for all classes",
                    "Single prediction per image",
                    "Faster inference time"
                ],
                "output_format": {
                    "predicted_class": "Single class prediction",
                    "confidence": "Prediction confidence",
                    "probabilities": "Dictionary of all class probabilities"
                },
                "model_path": CNN_MODEL_PATH if os.path.exists(CNN_MODEL_PATH) else None,
                "model_loaded": cnn_model is not None
            }
        },
        "comparison": {
            "yolo_advantages": [
                "Detects multiple objects",
                "Provides localization (bounding boxes)",
                "Better for overlapping objects"
            ],
            "cnn_advantages": [
                "Simpler architecture",
                "Faster inference for single objects",
                "Higher accuracy for image-level classification",
                "Less computational overhead"
            ],
            "use_cases": {
                "yolo": "When you need to locate and identify multiple leaves in an image",
                "cnn": "When you need to classify a single leaf image"
            }
        },
        "classes": {str(i): name for i, name in enumerate(cnn_classes)}
    })


# Base64 prediction endpoints
@app.route('/predict/yolo/base64', methods=['POST'])
def predict_yolo_base64():
    """YOLO prediction from base64 encoded image."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        image_data = base64.b64decode(data['image'])
        image_file = io.BytesIO(image_data)
        
        result, error = _predict_yolo_internal(image_file)
        
        if error:
            return jsonify({"error": f"YOLO prediction failed: {error}"}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"YOLO prediction failed: {str(e)}"}), 500


@app.route('/predict/cnn/base64', methods=['POST'])
def predict_cnn_base64():
    """CNN prediction from base64 encoded image."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        image_data = base64.b64decode(data['image'])
        image_file = io.BytesIO(image_data)
        
        result, error = _predict_cnn_internal(image_file)
        
        if error:
            return jsonify({"error": f"CNN prediction failed: {error}"}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"CNN prediction failed: {str(e)}"}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Leaf Detection & Classification API")
    print("YOLOv11 (Detection) + CNN (Classification)")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading models...")
    load_yolo_model()
    load_cnn_model()
    
    print("\n🚀 Starting Flask API Server...")
    print(f"   Server running at: http://localhost:5000")
    
    print("\n📋 Available Endpoints:")
    print("   GET  /              - Web frontend")
    print("   GET  /health        - Health check")
    print("   GET  /classes       - Get list of classes")
    print("   GET  /compare       - Compare models")
    print("   POST /predict/yolo  - YOLO detection")
    print("   POST /predict/cnn   - CNN classification")
    print("   POST /predict       - Both models (unified)")
    print("   POST /predict/yolo/base64  - YOLO (base64)")
    print("   POST /predict/cnn/base64   - CNN (base64)")
    
    print("\n📝 Example curl requests:")
    print('   curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict/yolo')
    print('   curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict/cnn')
    print('   curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict')
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("DEBUG", "False").lower() == "true"
    )
