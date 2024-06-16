import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Initialize the face detector and landmark detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_model = "lbfmodel_new.yaml/lbfmodel_new-b13371dce0ce924d3df83a9422113887.yaml"
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(landmark_model)

def detect_faces_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    
    _, landmarks = landmark_detector.fit(gray, faces)
    return landmarks[0], faces[0]

def compute_surface_normals(shape, image_size):
    h, w = image_size
    z = np.zeros((h, w))
    
    for (x, y) in shape[0]:
        x_int = int(round(x))
        y_int = int(round(y))
        z[y_int, x_int] = 1

    z = gaussian_filter(z, sigma=5)
    dzdx, dzdy = np.gradient(z)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(z)))
    norm = np.linalg.norm(normals, axis=2)
    normals /= np.expand_dims(norm, axis=2)
    return normals


def apply_directional_lighting(image, normals, light_direction):
    intensity = np.dot(normals, light_direction)
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    lighting_image = image.astype(np.float32) * intensity[:, :, np.newaxis]
    return np.clip(lighting_image, 0, 255).astype(np.uint8)

def extrapolate_modulation_factors(modulation, image_size):
    h, w = image_size
    known_values = modulation > 0
    laplace = csr_matrix((h * w, h * w))

    def index(y, x):
        return y * w + x

    for y in range(h):
        for x in range(w):
            if known_values[y, x]:
                laplace[index(y, x), index(y, x)] = 1
            else:
                laplace[index(y, x), index(y, x)] = -4
                if y > 0:
                    laplace[index(y, x), index(y - 1, x)] = 1
                if y < h - 1:
                    laplace[index(y, x), index(y + 1, x)] = 1
                if x > 0:
                    laplace[index(y, x), index(y, x - 1)] = 1
                if x < w - 1:
                    laplace[index(y, x), index(y, x + 1)] = 1

    b = np.zeros(h * w)
    for y in range(h):
        for x in range(w):
            if known_values[y, x].any():
                b[index(y, x)] = modulation[y, x]

    x = spsolve(laplace, b)
    modulation = x.reshape((h, w))
    return modulation

def render_lighting(image, shape, light_direction):
    normals = compute_surface_normals(shape, image.shape[:2])
    modulation = apply_directional_lighting(image, normals, light_direction)
    extrapolated_modulation = extrapolate_modulation_factors(modulation, image.shape[:2])
    output_image = image.astype(np.float32) * extrapolated_modulation[:, :, np.newaxis]
    return np.clip(output_image, 0, 255).astype(np.uint8)

# Test the function
image_path = "../single_face_test/test.jpg"
image = cv2.imread(image_path)

shape, rect = detect_faces_and_landmarks(image)
if shape is not None:
    light_direction = np.array([0, 0, 1])  # Change this to simulate different light directions
    output_image = render_lighting(image, shape, light_direction)
    cv2.imshow("Original", image)
    cv2.imshow("Relighted", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected")
