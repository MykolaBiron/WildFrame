def classify_animal(model, image_path):
    name = None
    results = model.predict(image_path)
    result = results[0]

    if len(result.boxes) > 0:
        box = result.boxes[0]
        class_id = int(box.cls[0])
        name = result.names[class_id]
    
    return name