# My Monitoring App

This project is a machine learning model monitoring application built with a Flask backend and a Next.js frontend. It provides functionalities to monitor model performance, manage datasets, and handle alerts related to model drift and data drift.

## Project Structure

```
my-monitoring-app
├── backend
│   ├── app.py                # Main entry point for the Flask backend application
│   ├── requirements.txt       # Python dependencies for the backend
│   ├── models.py              # Data models used in the application
│   ├── routes
│   │   ├── api.py            # API routes for handling requests
│   │   └── auth.py           # Authentication-related routes
│   ├── services
│   │   └── monitor.py        # Business logic for monitoring models
│   ├── templates
│   │   └── base.html         # Base HTML template for rendering views
│   └── static
│       ├── css               # CSS files for styling
│       └── js                # JavaScript files for client-side functionality
├── frontend
│   ├── package.json           # Configuration for the Next.js frontend
│   ├── next.config.js         # Configuration settings for Next.js
│   ├── pages
│   │   ├── index.js          # Main entry point for the Next.js application
│   │   ├── models.js         # Models overview page
│   │   └── alerts.js         # Alerts page
│   ├── components
│   │   ├── Dashboard.js      # Dashboard component displaying statistics
│   │   ├── ModelCard.js      # Component for displaying individual model information
│   │   └── AlertList.js      # Component for listing recent alerts
│   ├── styles
│   │   ├── globals.css       # Global CSS styles
│   │   └── dashboard.module.css # CSS module styles for the dashboard
├── .gitignore                 # Files and directories to be ignored by Git
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-monitoring-app
   ```

2. Set up the backend:
   - Navigate to the `backend` directory:
     ```
     cd backend
     ```
   - Install the required Python packages:
     ```
     pip install -r requirements.txt
     ```

3. Set up the frontend:
   - Navigate to the `frontend` directory:
     ```
     cd ../frontend
     ```
   - Install the required Node.js packages:
     ```
     npm install
     ```

## Usage

1. Start the Flask backend:
   ```
   cd backend
   python app.py
   ```

2. Start the Next.js frontend:
   ```
   cd frontend
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000` to access the application.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.