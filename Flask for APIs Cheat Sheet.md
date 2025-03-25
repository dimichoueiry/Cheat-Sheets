# Flask for APIs Cheat Sheet

## 1️⃣ Import Convention

```python
python
CopyEdit
from flask import Flask, request, jsonify, make_response
from flask_restful import Api, Resource
from flask_stateless import Stateless

```

---

## 2️⃣ Core Functions/Classes/Concepts Table:

| Function/Concept | Example Usage | Short Description |
| --- | --- | --- |
| `Flask(__name__)` | `app = Flask(__name__)` | Initializes Flask app |
| `@app.route()` | `@app.route('/hello', methods=['GET'])` | Defines a route and allowed HTTP methods |
| `request` | `data = request.get_json()` | Handles incoming request data (query params, JSON, headers) |
| `jsonify()` | `return jsonify({'result': 'success'})` | Converts Python dicts to JSON response |
| `make_response()` | `return make_response(jsonify({'error': 'Not found'}), 404)` | Custom response with status codes |
| `Api()` | `api = Api(app)` | Adds Flask-RESTful support |
| `Resource` | `class HelloWorld(Resource): ...` | Defines resources for RESTful APIs |
| `api.add_resource()` | `api.add_resource(HelloWorld, '/hello')` | Maps resource class to endpoint |
| `Stateless()` | `Stateless(app, secret='secret-key')` | Adds JWT authentication without sessions (via flask_stateless) |
| `@app.before_request` | `@app.before_request` → `def check(): pass` | Run function **before** each request |
| `@app.after_request` | `@app.after_request` → `def add_cors(response): pass` | Modify response after request (commonly for CORS headers) |
| `app.run()` | `app.run(debug=True)` | Runs the server |

---

## 3️⃣ Common Operations & Their Usage:

### 1. **Basic GET & POST Route**

```python
python
CopyEdit
@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        data = request.get_json()
        return jsonify({'received': data}), 201
    return jsonify({'message': 'Send a POST request!'}), 200

```

---

### 2. **Using Flask-RESTful Resource**

```python
python
CopyEdit
class Item(Resource):
    def get(self, name):
        return {'item': name}

    def post(self, name):
        data = request.get_json()
        return {'item': name, 'data': data}, 201

api.add_resource(Item, '/item/<string:name>')

```

---

### 3. **Handling Query Parameters & Headers**

```python
python
CopyEdit
@app.route('/query')
def query_example():
    name = request.args.get('name', 'Guest')
    user_agent = request.headers.get('User-Agent')
    return jsonify({'name': name, 'user_agent': user_agent})

```

---

### 4. **Adding JWT Auth via Flask-Stateless**

```python
python
CopyEdit
from flask_stateless import Stateless
from flask_stateless.decorators import jwt_required

Stateless(app, secret='my_secret')

@app.route('/secure')
@jwt_required
def secure_route():
    return jsonify({'message': 'Access granted!'})

```

---

### 5. **CORS Handling**

```python
python
CopyEdit
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

```

---

## 4️⃣ Useful Tips / Pro Tips / Best Practices:

Tip/Best Practice

---

**Always return JSON**

using

```
jsonify()
```

— never raw strings in APIs

---

Use

**blueprints**

for modularizing large API projects

---

Use

```
request.get_json(silent=True)
```

to avoid crashing if the body is not JSON

---

Set

```
debug=True
```

only during development —

**never in production**

---

**Status Codes matter!**

→ e.g.,

```
200 OK
```

,

```
201 Created
```

,

```
400 Bad Request
```

,

```
404 Not Found
```

---

**Flask-RESTful > Manual Routes**

for larger APIs → cleaner code, built-in parsing

---

**Input Validation:**

Use libraries like

**Marshmallow**

or

**Flask-Inputs**

---

For production, use a proper WSGI server (e.g.,

**Gunicorn**

) instead of

```
app.run()
```

---

---

## 5️⃣ (Optional) Integration for ML/DL:

**Example: Serving ML Models with Flask**

```python
python
CopyEdit
import joblib
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

```

**Possible Integrations:**

- **NumPy/Pandas:** Preprocess incoming JSON before passing to model.
- **Torch/TensorFlow:** Load model weights and infer.
- **Dockerize Flask App** for deployment.

---

## 6️⃣ Mini Project: **Build a Simple ML-Integrated Flask API**

### Tasks:

1. ✅ Create a Flask app.
2. ✅ Add a `/hello` route returning a JSON greeting.
3. ✅ Add a `/predict` route that:
    - Accepts POST requests.
    - Takes JSON input: `{ "features": [1, 2, 3, 4] }`.
    - Returns a dummy prediction (e.g., sum of features).
4. ✅ Use `make_response()` to return a 400 error if the input is invalid.
5. ✅ Add simple JWT authentication using `flask_stateless`.
6. ✅ Handle CORS properly.
7. ✅ Create a Flask-RESTful Resource to manage `/items` with GET & POST.
8. ✅ Add query parameter handling in one route.
9. ✅ Add a `before_request` logger printing the request path.
10. ✅ Package it, run it, and test with `curl` or Postman.
