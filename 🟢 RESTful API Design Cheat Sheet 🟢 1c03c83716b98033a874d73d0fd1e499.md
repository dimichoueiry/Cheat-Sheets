# 🟢 RESTful API Design Cheat Sheet 🟢

- 1
    
    ## Import Convention (Python Flask Example)
    
    ```python
    python
    CopyEdit
    from flask import Flask, request, jsonify
    from flask_restful import Resource, Api
    
    ```
    
    ---
    
    ## Core Functions/Classes/Concepts Table:
    
    | Function/Method/Concept Name | Example Usage | Short Description |
    | --- | --- | --- |
    | **HTTP Methods** | `GET`, `POST`, `PUT`, `DELETE` | Fundamental verbs for CRUD operations |
    | **URL Endpoint Design** | `/api/users/<int:id>` | Clean, hierarchical resource-based URLs |
    | **Status Codes** | `200 OK`, `201 Created`, `400 Bad Request`, `404 Not Found`, `500 Internal Server Error` | Standard codes indicating API response results |
    | **JSON Payloads** | `jsonify(data)` | REST APIs commonly return JSON |
    | **Idempotency** | `PUT`/`DELETE` requests | Same request produces same effect if repeated |
    | **Statelessness** | No session stored on server | Every request contains all necessary context |
    | **Versioning** | `/api/v1/resource` | Keep APIs backward-compatible |
    | **Authentication** | Token-Based, OAuth2, API Keys | Secure access control |
    | **Pagination** | `GET /items?page=2&limit=10` | Efficient large dataset handling |
    | **Filtering/Sorting** | `GET /items?sort=name&filter=status:active` | Client-controlled data querying |
    | **Error Handling** | `return {'error': 'Bad Request'}, 400` | Consistent, descriptive error messages |
    | **HATEOAS (Hypermedia as Engine of Application State)** | Add links in response | Guides client to next possible actions |
    
    ---
    
    ## Common Operations & Their Usage:
    
    ### 1. **Basic CRUD Operations (Flask-RESTful Example)**
    
    ```python
    python
    CopyEdit
    class User(Resource):
        def get(self, user_id):
            return {'user_id': user_id}, 200  # READ
    
        def post(self):
            data = request.get_json()
            return {'message': 'User created', 'data': data}, 201  # CREATE
    
        def put(self, user_id):
            data = request.get_json()
            return {'message': f'User {user_id} updated', 'data': data}, 200  # UPDATE
    
        def delete(self, user_id):
            return {'message': f'User {user_id} deleted'}, 204  # DELETE
    
    ```
    
    ### 2. **Pagination & Filtering**
    
    ```python
    python
    CopyEdit
    @app.route('/items')
    def get_items():
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=10, type=int)
        sort = request.args.get('sort', default='name')
        return jsonify({'page': page, 'limit': limit, 'sort_by': sort})
    
    ```
    
    ### 3. **Error Handling**
    
    ```python
    python
    CopyEdit
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    ```
    
    ### 4. **Authentication Example**
    
    ```python
    python
    CopyEdit
    from flask_httpauth import HTTPTokenAuth
    auth = HTTPTokenAuth(scheme='Bearer')
    
    @auth.verify_token
    def verify_token(token):
        return token == 'mysecrettoken'
    
    ```
    
    ---
    
    ## Useful Tips / Pro Tips / Best Practices:
    
    ✅ **Use Nouns, Not Verbs in URLs**
    
    Good: `/api/users`
    
    Bad: `/api/getUsers`
    
    ✅ **Consistent Naming Conventions**
    
    Stick to plural nouns for collections: `/users/`
    
    ✅ **Statelessness**
    
    Don't store session data on the server. Every request must contain necessary auth & context.
    
    ✅ **Version Your API Early**
    
    Use URI versioning like `/api/v1/` to avoid breaking clients when evolving.
    
    ✅ **Return Standard Status Codes**
    
    Don't return `200 OK` for errors—use `400`, `401`, `404`, `500`, etc.
    
    ✅ **Provide Clear, Consistent Error Messages**
    
    Include `error_code`, `message`, and helpful hints.
    
    ✅ **Document Your API (Swagger/OpenAPI)**
    
    Use tools like Swagger to auto-generate API docs.
    
    ✅ **Rate Limiting & Security**
    
    Prevent abuse with rate limiting, input validation, CORS policies, and secure tokens.
    
    ✅ **Support Filtering, Sorting, Pagination**
    
    Empower clients to query data efficiently.
    
    ✅ **Use JSON:API or similar spec if needed**
    
    It offers standardization in responses, relationships, errors, pagination, etc.
    
    ---
    
    ## Mini Project: Build a Mini "User Management API"
    
    Tasks:
    
    1. **Task 1:** Create a Flask-RESTful project with `/api/users/` endpoint.
    2. **Task 2:** Implement **GET**, **POST**, **PUT**, **DELETE** methods for `/api/users/`.
    3. **Task 3:** Add URL parameters to get a user by ID: `/api/users/<id>`.
    4. **Task 4:** Implement Pagination: Support `/api/users?page=1&limit=10`.
    5. **Task 5:** Add simple token-based authentication (e.g., hardcoded token).
    6. **Task 6:** Implement error handling for invalid user ID and return `404`.
    7. **Task 7:** Add filtering by username: `/api/users?filter=username:John`.
    8. **Task 8:** Return proper HTTP status codes (201 for create, 204 for delete, etc.).
    9. **Task 9:** Write clear, consistent JSON responses, including metadata.
    10. **Task 10 (Optional Advanced):** Add Swagger/OpenAPI docs using `flasgger` or `apispec`.
- 2
    
    # 🚀 API Design Principles (Focus on RESTful APIs)
    
    ---
    
    ## 🟢 What is an API?
    
    **API (Application Programming Interface)** is a set of rules & protocols that allows two software applications to communicate with each other.
    
    **Example:**
    
    When you use a weather app, the app communicates with a weather server API to fetch data like temperature, location, forecast, etc.
    
    ---
    
    ## 🟢 What is RESTful API?
    
    **REST (Representational State Transfer)** is an architectural style for designing APIs based on HTTP protocol.
    
    Key ideas:
    
    | Principle | Explanation |
    | --- | --- |
    | **Stateless** | No session info stored on server. Each request contains all context. |
    | **Client-Server** | Clear separation between client (frontend) & server (backend). |
    | **Uniform Interface** | Standardized way to interact (HTTP methods, URIs, etc.). |
    | **Cacheable** | Responses must define if they can be cached to improve performance. |
    | **Layered System** | APIs can be composed in layers, with each layer independent. |
    | **Code on Demand (Optional)** | Server can send executable code (e.g., JS) to client if needed. |
    
    ---
    
    ## 🟢 Core RESTful API Design Concepts:
    
    ### 1. **HTTP Methods (CRUD Mapping)**
    
    | HTTP Method | CRUD Operation | Usage Example |
    | --- | --- | --- |
    | `GET` | Read | Fetch a resource `/users/1` |
    | `POST` | Create | Create a resource `/users` |
    | `PUT` | Update (Replace) | Replace a resource `/users/1` |
    | `PATCH` | Update (Partial) | Modify part of resource `/users/1` |
    | `DELETE` | Delete | Delete resource `/users/1` |
    
    ---
    
    ### 2. **Resource-Based URIs**
    
    Use **nouns**, not verbs:
    
    ✅ Good:
    
    `/api/users`
    
    `/api/users/123/posts`
    
    ❌ Bad:
    
    `/api/getAllUsers`
    
    `/api/deleteUser`
    
    ---
    
    ### 3. **Versioning**
    
    Version your APIs early:
    
    ```
    bash
    CopyEdit
    /api/v1/users
    /api/v2/users
    
    ```
    
    This avoids breaking existing clients when you update functionality.
    
    ---
    
    ### 4. **Status Codes**
    
    Always return **standard HTTP status codes**:
    
    | Code | Meaning | When to Use |
    | --- | --- | --- |
    | 200 | OK | Successful GET/PUT request |
    | 201 | Created | Resource successfully created (POST) |
    | 204 | No Content | Successful DELETE with no body |
    | 400 | Bad Request | Client error (invalid input) |
    | 401 | Unauthorized | Authentication required |
    | 403 | Forbidden | Client not allowed |
    | 404 | Not Found | Resource doesn't exist |
    | 500 | Internal Server Error | Server-side issue |
    
    ---
    
    ### 5. **Pagination, Filtering, Sorting**
    
    - **Pagination:** Avoid loading too much data:
        
        ```
        bash
        CopyEdit
        GET /users?page=2&limit=50
        
        ```
        
    - **Filtering:**
        
        ```
        bash
        CopyEdit
        GET /users?status=active
        
        ```
        
    - **Sorting:**
        
        ```
        pgsql
        CopyEdit
        GET /users?sort=name&order=asc
        
        ```
        
    
    ---
    
    ### 6. **Statelessness**
    
    No session data is stored on the server.
    
    Each request **must contain authentication info & necessary context** (e.g., token in header).
    
    ---
    
    ### 7. **Authentication & Security**
    
    Common methods:
    
    | Type | Example | Usage |
    | --- | --- | --- |
    | API Key | `Authorization: ApiKey abc123` | Simple, but less secure |
    | Token-Based (JWT) | `Authorization: Bearer token` | Popular, stateless |
    | OAuth2 | OAuth flows (used by Google, Facebook APIs) | Secure, scalable for large apps |
    
    Always use **HTTPS**!
    
    ---
    
    ### 8. **Consistent Error Handling**
    
    Structure error responses clearly:
    
    ```json
    json
    CopyEdit
    {
      "error": {
        "code": 404,
        "message": "User not found",
        "details": "No user exists with ID 123"
      }
    }
    
    ```
    
    ---
    
    ### 9. **HATEOAS (Optional Advanced Concept)**
    
    Hypermedia links included in the response to guide client:
    
    ```json
    json
    CopyEdit
    {
      "user": { "id": 1, "name": "John" },
      "links": [
        { "rel": "self", "href": "/users/1" },
        { "rel": "posts", "href": "/users/1/posts" }
      ]
    }
    
    ```
    
    ---
    
    ## 🟢 Best Practices Summary:
    
    ✅ Use **Plural Nouns** for Collections: `/users/`, `/posts/`
    
    ✅ Always version your API
    
    ✅ Provide **clear, consistent status codes**
    
    ✅ Support filtering, sorting, and pagination
    
    ✅ Use **token-based authentication or OAuth2**
    
    ✅ Design responses in **JSON format**
    
    ✅ Document your API (Swagger/OpenAPI highly recommended)
    
    ✅ Always validate client input to avoid injection attacks
    
    ✅ Avoid overloading one endpoint (stick to REST principles)
    
    ---
    
    ## 🟢 RESTful API in Action (Quick Example - Flask)
    
    ```python
    python
    CopyEdit
    from flask import Flask, jsonify, request
    
    app = Flask(__name__)
    
    users = [{ 'id': 1, 'name': 'John Doe' }]
    
    @app.route('/api/v1/users', methods=['GET'])
    def get_users():
        return jsonify(users), 200
    
    @app.route('/api/v1/users/<int:user_id>', methods=['GET'])
    def get_user(user_id):
        user = next((u for u in users if u['id'] == user_id), None)
        if user:
            return jsonify(user), 200
        return jsonify({'error': 'User not found'}), 404
    
    @app.route('/api/v1/users', methods=['POST'])
    def create_user():
        data = request.get_json()
        new_user = {'id': len(users) + 1, 'name': data['name']}
        users.append(new_user)
        return jsonify(new_user), 201
    
    ```
    
- 3
    
    # 🚀 **RESTful API Design — Mock Interview Answer Sheet**
    
    ---
    
    ### 1️⃣ **What is REST?**
    
    - REST stands for **Representational State Transfer**.
    - It's an architectural style that uses standard HTTP methods.
    - Principles: **Stateless**, **Client-Server**, **Cacheable**, **Uniform Interface**, **Layered System**, optionally **Code on Demand**.
    
    ---
    
    ### 2️⃣ **Explain CRUD operations in REST.**
    
    | CRUD Operation | HTTP Method | Example |
    | --- | --- | --- |
    | Create | POST | `POST /users` |
    | Read | GET | `GET /users/1` |
    | Update | PUT/PATCH | `PUT /users/1` |
    | Delete | DELETE | `DELETE /users/1` |
    
    ---
    
    ### 3️⃣ **What does it mean that REST APIs are stateless?**
    
    - **Stateless** = Server doesn't store session info.
    - Each request carries all necessary context (e.g., authentication token, parameters).
    
    ---
    
    ### 4️⃣ **How do you design good URL endpoints?**
    
    - Use **nouns**, not verbs.
    - **Plural resources** for collections:
        - Good: `/api/v1/users/`, `/api/v1/users/1/posts`
        - Avoid: `/getAllUsers`, `/deleteUser`
    - Follow hierarchy.
    
    ---
    
    ### 5️⃣ **How do you handle versioning in APIs?**
    
    - Use URI versioning:
        
        `/api/v1/users`
        
    - Keeps backward compatibility as the API evolves.
    
    ---
    
    ### 6️⃣ **What are common HTTP status codes you use?**
    
    | Code | Meaning | When Used |
    | --- | --- | --- |
    | 200 | OK | Successful GET/PUT |
    | 201 | Created | POST success |
    | 204 | No Content | DELETE success |
    | 400 | Bad Request | Invalid input |
    | 401 | Unauthorized | Missing/Invalid auth |
    | 404 | Not Found | Resource doesn't exist |
    | 500 | Internal Server Error | Server error |
    
    ---
    
    ### 7️⃣ **How do you handle large datasets in REST APIs?**
    
    - **Pagination:**
        
        `GET /users?page=2&limit=50`
        
    - **Filtering:**
        
        `GET /users?status=active`
        
    - **Sorting:**
        
        `GET /users?sort=name&order=asc`
        
    
    ---
    
    ### 8️⃣ **What is idempotency? Which HTTP methods are idempotent?**
    
    - **Idempotency = multiple identical requests have same effect.**
    - **Idempotent Methods:**
        - `GET` (no change, safe)
        - `PUT`, `DELETE` (no side effects after first call)
    - `POST` is **not idempotent** (creates new resource each time).
    
    ---
    
    ### 9️⃣ **How do you secure a REST API?**
    
    - Use **HTTPS** for secure communication.
    - **Token-based authentication** (Bearer tokens, JWT).
    - **OAuth2** for delegated access.
    - Implement **rate limiting** and **input validation**.
    - Use **CORS policies**.
    
    ---
    
    ### 🔟 **How do you handle errors in REST APIs?**
    
    - Return standard HTTP status codes (e.g., 400, 404, 500).
    - Provide consistent JSON error messages:
    
    ```json
    json
    CopyEdit
    {
      "error": {
        "code": 404,
        "message": "User not found",
        "details": "No user exists with ID 123"
      }
    }
    
    ```
    
    ---
    
    ### 1️⃣1️⃣ **What tools do you use for documenting APIs?**
    
    - **Swagger/OpenAPI** for auto-generating interactive API docs.
    - Helps frontend teams & external users understand API endpoints.
    
    ---
    
    ### 1️⃣2️⃣ **Optional (ML roles): How would you expose a Machine Learning model with REST API?**
    
    - Use **Flask** or **FastAPI**.
    - Wrap the model in a `/predict` POST endpoint.
    - Accept input (JSON), process with model, return predictions.
    - Ensure:
        - Input validation.
        - Stateless design.
        - Clear status codes & error handling.
        - Secure access (token-based).
    
    ---
    
    # ✅ **Quick Final Pro Tip to Mention:**
    
    "I always focus on **clean, consistent, well-documented, and secure REST API design** — following best practices like versioning, proper status codes, pagination, and authentication to ensure scalability and maintainability."